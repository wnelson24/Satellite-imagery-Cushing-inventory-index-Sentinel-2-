# sentinel2_cushing_history_fixed.py

import ee
import requests
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# --- Initialize Earth Engine (must already be authenticated) ---
ee.Initialize(project='satellitecrude')

# Define Cushing, OK region (5 km buffer)
cushing = ee.Geometry.Point([-96.765, 35.985]).buffer(5000)

# How many past weeks to fetch
weeks_back = 8
results = []

for i in range(weeks_back):
    end = datetime.now(timezone.utc) - timedelta(days=7 * i)
    start = end - timedelta(days=7)

    # Sentinel-2 collection
    collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                  .filterBounds(cushing)
                  .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                  .sort("CLOUDY_PIXEL_PERCENTAGE"))

    # Check if collection has images
    if collection.size().getInfo() == 0:
        print(f"⚠️ No image for {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
        continue

    img = collection.first()

    # Get thumbnail URL
    url = img.getThumbURL({
        "region": cushing,
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "dimensions": 512
    })

    print(f"📡 Downloading {end.strftime('%Y-%m-%d')} → {url}")

    # Save image locally
    filename = f"cushing_{end.strftime('%Y%m%d')}.png"
    img_data = requests.get(url).content
    with open(filename, "wb") as f:
        f.write(img_data)

    # --- Process with OpenCV ---
    pic = cv2.imread(filename)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles (tanks)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=20, param1=50, param2=30,
                               minRadius=3, maxRadius=30)

    brightness_values = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            tank_brightness = cv2.mean(gray, mask=mask)[0]
            brightness_values.append(tank_brightness)

    if brightness_values:
        avg_brightness = np.mean(brightness_values)
        circle_count = len(brightness_values)
    else:
        avg_brightness = np.mean(gray)
        circle_count = 0

    # Brightness → inventory estimate
    norm = avg_brightness / 255
    log_b = np.log1p(norm * 100)
    est_inventory = 20 + (60 * (log_b / np.log1p(100)))  # range ~20–80 mbbl

    results.append({
        "date": end.strftime("%Y-%m-%d"),
        "brightness": avg_brightness,
        "estimated_inventory": est_inventory,
        "circles_detected": circle_count
    })

# Save results
df_new = pd.DataFrame(results)

try:
    df_old = pd.read_csv("cushing_realtime_forecast.csv")
    df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["date"])
except FileNotFoundError:
    df_all = df_new

df_all = df_all.sort_values("date")
df_all.to_csv("cushing_realtime_forecast.csv", index=False)

print("✅ Saved to cushing_realtime_forecast.csv")
print(df_all.tail())

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df_all["date"], df_all["estimated_inventory"], marker="o", label="Estimated Inventory (mbbl)")
plt.xticks(rotation=45)
plt.ylabel("Estimated Inventory (mbbl)")
plt.title("Cushing Crude Storage (Satellite Index)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

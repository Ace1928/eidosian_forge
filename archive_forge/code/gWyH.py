import pandas as pd
import numpy as np
import os

# Define the number of data points
num_data_points = 1000000

# Define categorical variables and possible values
categories = {
    "color": ["red", "blue", "green", "yellow", "black", None],
    "size": ["small", "medium", "large", None],
    "shape": ["circle", "square", "triangle", "rectangle", None],
    "material": ["wood", "metal", "plastic", None],
}

# Generate random data for each category, including the possibility of None values
data = {
    category: np.random.choice(
        values, num_data_points, p=[1 / len(values)] * len(values)
    )
    for category, values in categories.items()
}

# Add a numeric label column with random integers
data["label"] = np.random.randint(0, 100, num_data_points)

# Add additional numerical data
data["weight"] = np.random.uniform(
    1, 100, num_data_points
)  # Random weights between 1 and 100
data["price"] = np.random.uniform(
    5.0, 500.0, num_data_points
)  # Random prices between $5 and $500
data["age"] = np.random.randint(
    1, 15, num_data_points
)  # Random ages between 1 and 15 years

# Introduce missing values randomly in numerical columns
for col in ["weight", "price", "age"]:
    data[col][np.random.choice([True, False], num_data_points, p=[0.05, 0.95])] = None

# Create DataFrame
df = pd.DataFrame(data)

# Define the directory and file path
directory = "/home/lloyd/Downloads/exampledata"
file_path = os.path.join(directory, "example.csv")

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f"Data generated and saved to {file_path}")

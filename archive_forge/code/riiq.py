import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename="data_transformation.log", level=logging.INFO)

# Load the dataset
data = pd.read_csv("/home/lloyd/Downloads/exampledata/example.csv")
logging.info(f"Loaded dataset: {len(data)} rows, {len(data.columns)} columns")

# Perform data transformations
# Step 1: Remove duplicates
data.drop_duplicates(inplace=True)
logging.info(f"Removed duplicates: {len(data)} rows remaining")

# Step 2: Handle missing values
data.fillna(0, inplace=True)
logging.info("Filled missing values with 0")

# Step 3: Rename columns
data.rename(columns={"old_name": "new_name"}, inplace=True)
logging.info("Renamed column 'old_name' to 'new_name'")

# Step 4: Filter data
filtered_data = data[data["column"] > 10]
logging.info(f"Filtered data: {len(filtered_data)} rows remaining")

# Save the transformed dataset
filtered_data.to_csv("transformed_dataset.csv", index=False)
logging.info("Saved transformed dataset to 'transformed_dataset.csv'")

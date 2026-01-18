import pandas as pd
import os

# Load the dataset
data = pd.read_csv("dataset.csv")

# Define the partition column
partition_column = "date"

# Create a directory for partitioned data
partition_directory = "partitioned_data"
os.makedirs(partition_directory, exist_ok=True)

# Partition the data based on the partition column
for partition_value, partition_data in data.groupby(partition_column):
    partition_path = os.path.join(partition_directory, f"{partition_value}.csv")
    partition_data.to_csv(partition_path, index=False)

print("Data partitioning completed.")

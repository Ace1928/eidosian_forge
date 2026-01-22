import os
import dvc.api

# Initialize DVC
os.system("dvc init")

# Add the dataset to DVC
os.system("dvc add dataset.csv")

# Commit the changes
os.system("git add dataset.csv.dvc")
os.system('git commit -m "Add dataset"')

# Push the dataset to remote storage
os.system("dvc push")

# Retrieve the dataset version
dataset_version = dvc.api.get_url("dataset.csv")
print(f"Dataset Version: {dataset_version}")

# Checkout a specific version of the dataset
os.system("git checkout <commit-hash>")
os.system("dvc checkout")

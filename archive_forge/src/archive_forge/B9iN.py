import os
import zipfile

# Define the dataset file
dataset_file = "dataset.csv"

# Create a ZIP archive
archive_name = "dataset_archive.zip"
with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(dataset_file)

# Move the dataset file to an archive directory
archive_directory = "archive"
os.makedirs(archive_directory, exist_ok=True)
os.rename(dataset_file, os.path.join(archive_directory, dataset_file))

print(f"Dataset archived: {archive_name}")

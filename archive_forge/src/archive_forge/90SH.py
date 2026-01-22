import boto3
import os

# Define the dataset file
dataset_file = "dataset.csv"

# Set up AWS S3 client
s3 = boto3.client("s3")

# Define the S3 bucket and object key
bucket_name = "your-bucket-name"
object_key = "backups/dataset.csv"

# Upload the dataset file to S3
s3.upload_file(dataset_file, bucket_name, object_key)

print(f"Dataset backed up to S3: s3://{bucket_name}/{object_key}")

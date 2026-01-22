import os
import logging
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define directories
RAW_DATA_DIR = "/media/lloyd/Aurora_M2/indegodata/raw_datasets"
PROCESSED_DATA_DIR = "/media/lloyd/Aurora_M2/indegodata/processed_data"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Datasets from Huggingface for training and testing models
HUGGINGFACE_URLS = [
    "https://huggingface.co/datasets/ambig_qa",
    "https://huggingface.co/datasets/break_data",
    "https://huggingface.co/datasets/tau/commonsense_qa",
    "https://huggingface.co/datasets/stanfordnlp/coqa",
    "https://huggingface.co/datasets/ucinlp/drop",
    "https://huggingface.co/datasets/hotpot_qa",
    "https://huggingface.co/datasets/narrativeqa",
    "https://huggingface.co/datasets/allenai/openbookqa",
    "https://huggingface.co/datasets/allenai/qasc",
    "https://huggingface.co/datasets/quac",
    "https://huggingface.co/datasets/rajpurkar/squad_v2",
    "https://huggingface.co/datasets/trec",
    "https://huggingface.co/datasets/wiki_qa",
    "https://huggingface.co/datasets/conv_ai",
]


# Function to download and process Huggingface datasets
def process_huggingface_datasets(urls, raw_data_dir, processed_data_dir):
    universal_train = []
    universal_test = []
    universal_validation = []

    for url in tqdm(urls, desc="Processing Huggingface datasets"):
        try:
            dataset_name = url.split("/")[-1]
            logging.info(f"Loading dataset: {dataset_name}")

            # Get all configurations for the dataset
            configs = get_dataset_config_names(dataset_name)
            for config in configs:
                logging.info(f"Processing config: {config}")

                # Get all splits for the configuration
                splits = get_dataset_split_names(dataset_name, config)
                for split in splits:
                    logging.info(f"Processing split: {split}")

                    # Check if the dataset split has already been processed
                    processed_split_dir = os.path.join(
                        processed_data_dir, dataset_name, config
                    )
                    if os.path.exists(
                        os.path.join(processed_split_dir, f"{split}.csv")
                    ):
                        logging.info(
                            f"Dataset {dataset_name} config {config} split {split} already processed. Skipping."
                        )
                        continue

                    # Load the dataset with the specific config and split
                    dataset = load_dataset(dataset_name, config, split=split)

                    # Create directories for raw and processed data
                    raw_split_dir = os.path.join(
                        raw_data_dir, dataset_name, config, split
                    )
                    os.makedirs(raw_split_dir, exist_ok=True)
                    os.makedirs(processed_split_dir, exist_ok=True)

                    # Save raw dataset to disk
                    dataset.save_to_disk(raw_split_dir)

                    # Convert to DataFrame and save
                    df = dataset.to_pandas()
                    df.to_csv(
                        os.path.join(processed_split_dir, f"{split}.csv"),
                        index=False,
                    )

                    # Append to the appropriate universal list
                    if split == "train":
                        universal_train.append(df)
                    elif split == "test":
                        universal_test.append(df)
                    elif split == "validation":
                        universal_validation.append(df)

        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}: {e}")

    # Save the merged universal datasets
    if universal_train:
        universal_train_df = pd.concat(universal_train)
        universal_train_df.to_csv(
            os.path.join(processed_data_dir, "universal_train.csv"), index=False
        )
    if universal_test:
        universal_test_df = pd.concat(universal_test)
        universal_test_df.to_csv(
            os.path.join(processed_data_dir, "universal_test.csv"), index=False
        )
    if universal_validation:
        universal_validation_df = pd.concat(universal_validation)
        universal_validation_df.to_csv(
            os.path.join(processed_data_dir, "universal_validation.csv"), index=False
        )


# Huggingface Processor
def huggingface_processor():
    process_huggingface_datasets(HUGGINGFACE_URLS, RAW_DATA_DIR, PROCESSED_DATA_DIR)


# Execute processor
huggingface_processor()

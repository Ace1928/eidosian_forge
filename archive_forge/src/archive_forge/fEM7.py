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
RAW_DATA_DIR = "/home/lloyd/Downloads/indegodata/raw_datasets"
PROCESSED_DATA_DIR = "/home/lloyd/Downloads/indegodata/processed_data"

# Datasets from Huggingface for training and testing models
HUGGINGFACE_URLS = [
    "https://huggingface.co/datasets/ambig_qa",
    "https://huggingface.co/datasets/break_data",
    "https://huggingface.co/datasets/tau/commonsense_qa",
    "https://huggingface.co/datasets/stanfordnlp/coqa",
    "https://huggingface.co/datasets/ucinlp/drop",
    "https://huggingface.co/datasets/HongzheBi/DuReader2.0",
    "https://huggingface.co/datasets/hotpot_qa",
    "https://huggingface.co/datasets/narrativeqa",
    "https://huggingface.co/datasets/natural_questions",
    "https://huggingface.co/datasets/newsqa",
    "https://huggingface.co/datasets/allenai/openbookqa",
    "https://huggingface.co/datasets/allenai/qasc",
    "https://huggingface.co/datasets/quac",
    "https://huggingface.co/datasets/rajpurkar/squad_v2",
    "https://huggingface.co/datasets/trec",
    "https://huggingface.co/datasets/tydiqa",
    "https://huggingface.co/datasets/wiki_qa",
    "https://huggingface.co/datasets/ubuntu_dialogs_corpus",
    "https://huggingface.co/datasets/conv_ai",
]


# Function to download and process Huggingface datasets
def process_huggingface_datasets(urls, raw_data_dir, processed_data_dir):
    universal_datasets = {}

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

                    # Load the dataset with the specific config and split
                    dataset = load_dataset(dataset_name, config, split=split)
                    dataset.save_to_disk(
                        os.path.join(raw_data_dir, f"{dataset_name}_{config}_{split}")
                    )

                    # Convert to DataFrame and save
                    df = dataset.to_pandas()
                    df.to_csv(
                        os.path.join(
                            processed_data_dir, f"{dataset_name}_{config}_{split}.csv"
                        ),
                        index=False,
                    )

                    # Merge datasets with the same config and split
                    key = f"{dataset_name}_{config}"
                    if key not in universal_datasets:
                        universal_datasets[key] = df
                    else:
                        universal_datasets[key] = pd.concat(
                            [universal_datasets[key], df]
                        )

        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}: {e}")

    # Save the merged universal datasets
    for key, df in universal_datasets.items():
        df.to_csv(os.path.join(processed_data_dir, f"{key}_universal.csv"), index=False)


# Huggingface Processor
def huggingface_processor():
    process_huggingface_datasets(HUGGINGFACE_URLS, RAW_DATA_DIR, PROCESSED_DATA_DIR)


# Execute processor
huggingface_processor()

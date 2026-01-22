import os
import logging
from convokit import Corpus, TextParser, PolitenessStrategies, download
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define directories
RAW_DATA_DIR = "/media/lloyd/Aurora_M2/indegodata/raw_datasets"
PROCESSED_DATA_DIR = "/media/lloyd/Aurora_M2/indegodata/processed_data"

# List of datasets to download from ConvoKit
CONVOKIT_DATASETS = [
    "supreme-corpus",
    "wiki-corpus",
    "reddit-corpus-small",
    "chromium-corpus",
    "winning-args-corpus",
    "reddit-coarse-discourse-corpus",
    "persuasionforgood-corpus",
    "iq2-corpus",
    "friends-corpus",
    "switchboard-corpus",
    "wikipedia-politeness-corpus",
    "stack-exchange-politeness-corpus",
    "diplomacy-corpus",
    "gap-corpus",
    "casino-corpus",
]


def process_convokit_datasets(
    datasets: list, raw_data_dir: str, processed_data_dir: str
) -> Corpus:
    """
    Download and process ConvoKit datasets, merging them into a universal corpus.

    Args:
        datasets (list): List of dataset names to download.
        raw_data_dir (str): Directory to store raw datasets.
        processed_data_dir (str): Directory to store processed datasets.

    Returns:
        Corpus: The merged universal corpus.
    """
    corpora = {}
    for dataset in tqdm(datasets, desc="Processing ConvoKit datasets"):
        processed_dataset_path = os.path.join(processed_data_dir, dataset)
        if os.path.exists(processed_dataset_path):
            logging.info(f"Dataset {dataset} already processed. Skipping download.")
            corpus = Corpus(filename=processed_dataset_path)
            corpora[dataset] = corpus
            continue

        try:
            logging.info(f"Downloading dataset: {dataset}")
            corpus = Corpus(filename=download(dataset, data_dir=raw_data_dir))
            corpora[dataset] = corpus
            corpus.dump(processed_dataset_path)
        except Exception as e:
            logging.error(f"Failed to download dataset {dataset}: {e}")

    if not corpora:
        raise ValueError("No datasets were successfully processed.")

    universal_corpus = list(corpora.values())[0]
    for corpus in list(corpora.values())[1:]:
        universal_corpus = universal_corpus.merge(corpus)

    universal_corpus.dump(os.path.join(processed_data_dir, "convokit_corpus"))
    return universal_corpus


def apply_transformers(corpus: Corpus) -> Corpus:
    """
    Apply text parsing and politeness strategies transformers to a corpus.

    Args:
        corpus (Corpus): The corpus to transform.

    Returns:
        Corpus: The transformed corpus.
    """
    parser = TextParser()
    politeness = PolitenessStrategies()
    logging.info("Applying transformers to corpus")
    corpus = parser.transform(corpus)
    corpus = politeness.transform(corpus)
    return corpus


def corpus_to_dataframes(corpus: Corpus, prefix: str, processed_data_dir: str) -> tuple:
    """
    Convert corpus components to DataFrames and save them as CSV files.

    Args:
        corpus (Corpus): The corpus to convert.
        prefix (str): Prefix for the CSV filenames.
        processed_data_dir (str): Directory to save the CSV files.

    Returns:
        tuple: DataFrames for utterances, conversations, and speakers.
    """
    try:
        utterances_df = corpus.get_utterances_dataframe()
        conversations_df = corpus.get_conversations_dataframe()
        speakers_df = corpus.get_speakers_dataframe()

        utterances_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_utterances.csv"), index=False
        )
        conversations_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_conversations.csv"), index=False
        )
        speakers_df.to_csv(
            os.path.join(processed_data_dir, f"{prefix}_speakers.csv"), index=False
        )

        return utterances_df, conversations_df, speakers_df
    except Exception as e:
        logging.error(f"Failed to convert corpus to DataFrames: {e}")
        raise


def convokit_processor():
    """
    Main processor function to handle the entire ConvoKit dataset processing pipeline.
    """
    convokit_corpus = process_convokit_datasets(
        CONVOKIT_DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
    )
    convokit_corpus = apply_transformers(convokit_corpus)
    convokit_utterances_df, convokit_conversations_df, convokit_speakers_df = (
        corpus_to_dataframes(convokit_corpus, "convokit", PROCESSED_DATA_DIR)
    )
    logging.info("Displaying the first few rows of the ConvoKit utterances DataFrame")
    print(convokit_utterances_df.head())


if __name__ == "__main__":
    convokit_processor()

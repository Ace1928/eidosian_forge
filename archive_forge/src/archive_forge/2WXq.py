import os
import logging
from datasets import load_dataset
from convokit import Corpus, TextParser, PolitenessStrategies, download
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


# Function to download and process Huggingface datasets
def process_huggingface_datasets(urls, raw_data_dir, processed_data_dir):
    corpora = {}
    for url in tqdm(urls, desc="Processing Huggingface datasets"):
        try:
            dataset_name = url.split("/")[-1]
            logging.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            dataset.save_to_disk(os.path.join(raw_data_dir, dataset_name))

            # Convert to ConvoKit corpus
            convokit_corpus = Corpus.from_dataset(dataset)
            convokit_corpus.dump(os.path.join(processed_data_dir, dataset_name))
            corpora[dataset_name] = convokit_corpus
        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}: {e}")

    # Merging datasets into a universal corpus
    universal_corpus = corpora[list(corpora.keys())[0]]
    for dataset_name in list(corpora.keys())[1:]:
        universal_corpus = universal_corpus.merge(corpora[dataset_name])

    # Save the merged corpus
    universal_corpus.dump(os.path.join(processed_data_dir, "huggingface_corpus"))

    return universal_corpus


# Function to download and process ConvoKit datasets
def process_convokit_datasets(datasets, raw_data_dir, processed_data_dir):
    corpora = {}
    for dataset in tqdm(datasets, desc="Processing ConvoKit datasets"):
        try:
            logging.info(f"Downloading dataset: {dataset}")
            corpora[dataset] = Corpus(filename=download(dataset, data_dir=raw_data_dir))
        except Exception as e:
            logging.error(f"Failed to download dataset {dataset}: {e}")

    # Merging datasets into a universal corpus
    universal_corpus = corpora[list(corpora.keys())[0]]
    for dataset in list(corpora.keys())[1:]:
        universal_corpus = universal_corpus.merge(corpora[dataset])

    # Save the merged corpus
    universal_corpus.dump(os.path.join(processed_data_dir, "convokit_corpus"))

    return universal_corpus


# Function to apply transformers to a corpus
def apply_transformers(corpus):
    parser = TextParser()
    politeness = PolitenessStrategies()
    logging.info("Applying transformers to corpus")
    corpus = parser.transform(corpus)
    corpus = politeness.transform(corpus)
    return corpus


# Converting corpus components to DataFrames
def corpus_to_dataframes(corpus, prefix, processed_data_dir):
    try:
        utterances_df = corpus.get_utterances_dataframe()
        conversations_df = corpus.get_conversations_dataframe()
        speakers_df = corpus.get_speakers_dataframe()

        # Save the dataframes to CSV for further use
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


# Huggingface Processor
def huggingface_processor():
    huggingface_corpus = process_huggingface_datasets(
        HUGGINGFACE_URLS, RAW_DATA_DIR, PROCESSED_DATA_DIR
    )
    huggingface_corpus = apply_transformers(huggingface_corpus)
    huggingface_utterances_df, huggingface_conversations_df, huggingface_speakers_df = (
        corpus_to_dataframes(huggingface_corpus, "huggingface", PROCESSED_DATA_DIR)
    )
    logging.info(
        "Displaying the first few rows of the Huggingface utterances DataFrame"
    )
    print(huggingface_utterances_df.head())


# ConvoKit Processor
def convokit_processor():
    convokit_corpus = process_convokit_datasets(
        CONVOKIT_DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
    )
    convokit_corpus = apply_transformers(convokit_corpus)
    convokit_utterances_df, convokit_conversations_df, convokit_speakers_df = (
        corpus_to_dataframes(convokit_corpus, "convokit", PROCESSED_DATA_DIR)
    )
    logging.info("Displaying the first few rows of the ConvoKit utterances DataFrame")
    print(convokit_utterances_df.head())


# Execute processors
huggingface_processor()
convokit_processor()

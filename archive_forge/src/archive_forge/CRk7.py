# Import the os module, which provides a way of using operating system dependent functionality like reading or writing to the file system.
import os

# Import the torch module, which is the main library for PyTorch, an open-source machine learning library.
import torch

# Import the PreTrainedTokenizerFast class from the transformers library, which provides fast tokenization for pre-trained models.
from transformers import PreTrainedTokenizerFast

# Import various classes and functions from the tokenizers library, which is used for building and training tokenizers.
from tokenizers import (
    Tokenizer,  # Import the Tokenizer class, which is the main class for tokenization.
    models,  # Import the models module, which contains different tokenization models.
    normalizers,  # Import the normalizers module, which contains text normalization functions.
    pre_tokenizers,  # Import the pre_tokenizers module, which contains pre-tokenization functions.
    processors,  # Import the processors module, which contains post-processing functions.
    trainers,  # Import the trainers module, which contains training functions for tokenizers.
)

# Import specific normalizers from the tokenizers.normalizers module.
from tokenizers.normalizers import NFD, Lowercase, StripAccents

# Import specific pre-tokenizers from the tokenizers.pre_tokenizers module.
from tokenizers.pre_tokenizers import Whitespace, Digits

# Import the SentenceTransformer class from the sentence_transformers library, which is used for sentence embeddings.
from sentence_transformers import SentenceTransformer

# Import the nltk module, which is the Natural Language Toolkit, a library for working with human language data.
import nltk

# Import the SentimentIntensityAnalyzer class from the nltk.sentiment.vader module, which is used for sentiment analysis.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import the word_tokenize function from the nltk.tokenize module, which is used for tokenizing text into words.
from nltk.tokenize import word_tokenize

# Import the stopwords and wordnet modules from the nltk.corpus package, which provide access to stopwords and WordNet lexical database.
from nltk.corpus import stopwords, wordnet

# Import the pos_tag function and ne_chunk function from the nltk module, which are used for part-of-speech tagging and named entity recognition.
from nltk import pos_tag, ne_chunk

# Import the DataLoader and Dataset classes from the torch.utils.data module, which are used for loading and managing datasets in PyTorch.
from torch.utils.data import DataLoader, Dataset

# Import everything (*) from the tkinter module, which is used for creating graphical user interfaces.
from tkinter import *

# Import specific functions from the tkinter module for file dialogs and message boxes.
from tkinter import filedialog, messagebox

# Import the threading module, which is used for creating and managing threads.
import threading

# Import the logging module, which provides a flexible framework for emitting log messages from Python programs.
import logging

# Configure the logging module to display log messages with a specific format and level.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define a function to download necessary NLTK resources for sentiment analysis, tokenization, stopwords, part-of-speech tagging, named entity recognition, and WordNet.
def download_nltk_resources():
    logging.info("Starting download of NLTK resources.")
    # List of required NLTK resources.
    resources = [
        "vader_lexicon",
        "punkt",
        "stopwords",
        "averaged_perceptron_tagger",
        "wordnet",
        "maxent_ne_chunker",
        "words",
        "omw",
        "universal_tagset",
    ]
    # Set the NLTK data path to a specific directory.
    nltk.data.path.append("/home/lloyd/Downloads/nltk-data")
    logging.info("NLTK data path set to /home/lloyd/Downloads/nltk-data.")
    # Iterate over each resource in the list.
    for resource in resources:
        try:
            # Check if the resource is already downloaded.
            nltk.data.find(f"tokenizers/{resource}")
            logging.info(f"Resource '{resource}' already downloaded.")
        except LookupError:
            # Download the resource if it is not found.
            logging.info(f"Resource '{resource}' not found. Downloading...")
            nltk.download(resource, download_dir="/home/lloyd/Downloads/nltk-data")
            logging.info(f"Resource '{resource}' downloaded successfully.")
    logging.info("Completed download of NLTK resources.")


# Call the function to download NLTK resources.
download_nltk_resources()


# Define a custom dataset class named TextDataset that inherits from the Dataset class in PyTorch.
class TextDataset(Dataset):
    # Initialize the TextDataset class with texts, tokenizer, and max_length parameters.
    def __init__(self, texts, tokenizer, max_length=512):
        logging.info("Initializing TextDataset.")
        # Store the texts parameter as an instance variable.
        self.texts = texts
        # Store the tokenizer parameter as an instance variable.
        self.tokenizer = tokenizer
        # Store the max_length parameter as an instance variable with a default value of 512.
        self.max_length = max_length
        logging.info(
            f"TextDataset initialized with {len(texts)} texts, max_length={max_length}."
        )

    # Define the __len__ method to return the number of texts in the dataset.
    def __len__(self):
        return len(self.texts)

    # Define the __getitem__ method to get a specific item from the dataset by index.
    def __getitem__(self, idx):
        logging.debug(f"Fetching item at index {idx}.")
        # Get the text at the specified index.
        text = self.texts[idx]
        # Tokenize the text using the tokenizer, with specified max_length, padding, truncation, and return_tensors options.
        try:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            logging.debug(f"Tokenization successful for index {idx}.")
            # Return the input IDs and attention mask as tensors, squeezed to remove single-dimensional entries.
            return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()
        except Exception as e:
            logging.error(f"Error tokenizing text at index {idx}: {e}", exc_info=True)
            return None, None


# Define a function named load_data_from_directory to load text data from a specified directory.
def load_data_from_directory(directory_path):
    logging.info(f"Loading data from directory: {directory_path}")
    # Initialize an empty list to store the texts.
    texts = []
    try:
        # Iterate over the filenames in the specified directory.
        for filename in os.listdir(directory_path):
            # Check if the filename ends with ".txt".
            if filename.endswith(".txt"):
                logging.debug(f"Reading file: {filename}")
                # Open the file in read mode with UTF-8 encoding.
                with open(
                    os.path.join(directory_path, filename), "r", encoding="utf-8"
                ) as file:
                    # Read the contents of the file and append it to the texts list.
                    texts.append(file.read())
        logging.info(f"Loaded {len(texts)} text files from directory.")
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
    # Return the list of texts.
    return texts


# Define a function named initialize_and_train_tokenizer to initialize and train a tokenizer on a list of texts.
def initialize_and_train_tokenizer(texts, vocab_size=50000, special_tokens=None):
    logging.info("Initializing and training tokenizer.")
    if special_tokens is None:
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    try:
        # Create a new Tokenizer object with a Byte-Pair Encoding (BPE) model.
        tokenizer = Tokenizer(models.BPE())
        logging.info("Tokenizer with BPE model created successfully.")

        # Set the normalizer for the tokenizer to a sequence of NFD, Lowercase, and StripAccents normalizers.
        tokenizer.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()]
        )
        logging.info("Normalizer set to sequence of NFD, Lowercase, and StripAccents.")

        # Set the pre-tokenizer for the tokenizer to a sequence of Whitespace and Digits pre-tokenizers.
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [Whitespace(), Digits(individual_digits=True)]
        )
        logging.info("Pre-tokenizer set to sequence of Whitespace and Digits.")

        # Create a BpeTrainer object with specified special tokens and vocabulary size.
        trainer = trainers.BpeTrainer(
            special_tokens=special_tokens, vocab_size=vocab_size
        )
        logging.info(
            f"BpeTrainer created with special tokens: {special_tokens} and vocab size: {vocab_size}."
        )

        # Train the tokenizer on the list of texts using the BpeTrainer.
        tokenizer.train_from_iterator(texts, trainer=trainer)
        logging.info("Tokenizer training completed successfully.")

        # Set the post-processor for the tokenizer to a TemplateProcessing object with specified templates and special tokens.
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        logging.info("Post-processor set with specified templates and special tokens.")

        # Wrap the tokenizer in a PreTrainedTokenizerFast object with specified special tokens.
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        logging.info("Tokenizer wrapped in PreTrainedTokenizerFast successfully.")

        # Return the wrapped tokenizer.
        return wrapped_tokenizer
    except Exception as e:
        logging.error(f"Error initializing and training tokenizer: {e}", exc_info=True)
        return None


# Initialize a SentenceTransformer object with the "all-MiniLM-L6-v2" model for semantic analysis.
try:
    logging.info("Initializing SentenceTransformer with model 'all-MiniLM-L6-v2'.")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("SentenceTransformer initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing SentenceTransformer: {e}", exc_info=True)
    semantic_model = None

# Initialize a SentimentIntensityAnalyzer object for sentiment analysis.
try:
    logging.info("Initializing SentimentIntensityAnalyzer.")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    logging.info("SentimentIntensityAnalyzer initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing SentimentIntensityAnalyzer: {e}", exc_info=True)
    sentiment_analyzer = None


# Define a function named advanced_tokenize to extend tokenization with semantic and sentiment analysis.
def advanced_tokenize(text, tokenizer):
    logging.info(f"Starting advanced tokenization for text: {text[:30]}...")
    try:
        # Tokenize the text using the tokenizer.
        tokens = tokenizer.tokenize(text)
        logging.debug(f"Tokens: {tokens}")
        # Encode the tokens into embeddings using the semantic model, converting the result to a tensor.
        embeddings = semantic_model.encode(tokens, convert_to_tensor=True)
        logging.debug(f"Embeddings: {embeddings}")
        # Analyze the sentiment of each token using the sentiment analyzer and store the compound scores in a list.
        sentiments = [
            sentiment_analyzer.polarity_scores(token)["compound"] for token in tokens
        ]
        logging.debug(f"Sentiments: {sentiments}")
        # Return the tokens, embeddings, and sentiments.
        logging.info("Advanced tokenization completed successfully.")
        return tokens, embeddings, sentiments
    except Exception as e:
        logging.error(f"Error in advanced tokenization: {e}", exc_info=True)
        return None, None, None


# Define a function named comprehensive_nlp_analysis to perform comprehensive NLP analysis on a text.
def comprehensive_nlp_analysis(text):
    logging.info(f"Starting comprehensive NLP analysis for text: {text[:30]}...")
    try:
        # Tokenize the text into words using the word_tokenize function.
        tokens = word_tokenize(text)
        logging.debug(f"Tokens: {tokens}")
        # Get the set of English stopwords.
        stop_words = set(stopwords.words("english"))
        logging.debug(f"Stop words: {stop_words}")
        # Filter out the stopwords from the tokens.
        filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
        logging.debug(f"Filtered tokens: {filtered_tokens}")
        # Perform part-of-speech tagging on the filtered tokens.
        pos_tags = pos_tag(filtered_tokens)
        logging.debug(f"POS tags: {pos_tags}")
        # Perform named entity recognition on the part-of-speech tagged tokens.
        named_entities = ne_chunk(pos_tags)
        logging.debug(f"Named entities: {named_entities}")
        # Lemmatize the filtered tokens using WordNet.
        lemmas = [
            wordnet.morphy(token) if wordnet.morphy(token) else token
            for token in filtered_tokens
        ]
        logging.debug(f"Lemmas: {lemmas}")
        # Return a dictionary containing the tokens, filtered tokens, part-of-speech tags, named entities, and lemmas.
        logging.info("Comprehensive NLP analysis completed successfully.")
        return {
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "pos_tags": pos_tags,
            "named_entities": named_entities,
            "lemmas": lemmas,
        }
    except Exception as e:
        logging.error(f"Error in comprehensive NLP analysis: {e}", exc_info=True)
        return None


# Define a function named batch_tokenize to perform batch processing of texts.
def batch_tokenize(texts, tokenizer, batch_size=32):
    logging.info(
        f"Starting batch tokenization for {len(texts)} texts with batch size {batch_size}."
    )
    try:
        # Split the texts into batches of the specified batch size.
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logging.debug(f"Created {len(batches)} batches.")
        # Initialize empty lists to store the tokens, embeddings, sentiments, and NLP analyses for all batches.
        all_tokens, all_embeddings, all_sentiments, all_nlp_analyses = [], [], [], []
        # Iterate over each batch of texts.
        for batch in batches:
            logging.debug(f"Processing batch with {len(batch)} texts.")
            # Tokenize the batch of texts using the tokenizer with padding, truncation, and return_tensors options.
            tokens_batch = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            logging.debug(f"Tokens batch: {tokens_batch}")
            # Encode the batch of texts into embeddings using the semantic model, converting the result to a tensor.
            embeddings_batch = semantic_model.encode(batch, convert_to_tensor=True)
            logging.debug(f"Embeddings batch: {embeddings_batch}")
            # Analyze the sentiment of each text in the batch using the sentiment analyzer and store the compound scores in a list.
            sentiments_batch = [
                sentiment_analyzer.polarity_scores(text)["compound"] for text in batch
            ]
            logging.debug(f"Sentiments batch: {sentiments_batch}")
            # Perform comprehensive NLP analysis on each text in the batch.
            nlp_analyses_batch = [comprehensive_nlp_analysis(text) for text in batch]
            logging.debug(f"NLP analyses batch: {nlp_analyses_batch}")
            # Append the tokens, embeddings, sentiments, and NLP analyses for the batch to the respective lists.
            all_tokens.append(tokens_batch)
            all_embeddings.append(embeddings_batch)
            all_sentiments.append(sentiments_batch)
            all_nlp_analyses.append(nlp_analyses_batch)
        # Return the tokens, embeddings, sentiments, and NLP analyses for all batches.
        logging.info("Batch tokenization completed successfully.")
        return all_tokens, all_embeddings, all_sentiments, all_nlp_analyses
    except Exception as e:
        logging.error(f"Error in batch tokenization: {e}", exc_info=True)
        return None, None, None, None


# Define a function named load_data_thread to load data and process it in a separate thread.
def load_data_thread():
    logging.info("Starting load data thread.")
    try:
        # Open a file dialog to select a directory with text files and store the selected directory path.
        directory_path = filedialog.askdirectory(
            title="Select Directory with Text Files"
        )
        logging.info(f"Directory selected: {directory_path}")
        # Check if a directory path was selected.
        if directory_path:
            # Load the texts from the selected directory.
            texts = load_data_from_directory(directory_path)
            # Initialize and train a tokenizer on the loaded texts.
            tokenizer = initialize_and_train_tokenizer(texts)
            # Perform batch tokenization, embedding, sentiment analysis, and NLP analysis on the loaded texts.
            tokens, embeddings, sentiments, nlp_analyses = batch_tokenize(
                texts, tokenizer
            )
            # Iterate over each text and its corresponding analysis results.
            for i, text in enumerate(texts):
                # Print the text number and the text itself.
                logging.info(f"Text {i+1}: {text}")
                # Print the tokens for the text.
                logging.info(f"Tokens: {tokens[i]}")
                # Print the embeddings for the text.
                logging.info(f"Embeddings: {embeddings[i]}")
                # Print the sentiment scores for the text.
                logging.info(f"Sentiments: {sentiments[i]}")
                # Print the comprehensive NLP analysis results for the text.
                logging.info(f"NLP Analysis: {nlp_analyses[i]}")
    except Exception as e:
        logging.error(f"Error in load data thread: {e}", exc_info=True)


# Check if the script is being run as the main module.
if __name__ == "__main__":
    logging.info("Script started as main module.")
    try:
        # Create a Tkinter root window.
        root = Tk()
        logging.info("Tkinter root window created.")
        # Withdraw the root window to hide it.
        root.withdraw()
        logging.info("Tkinter root window withdrawn.")
        # Start a new thread to load data and process it using the load_data_thread function.
        threading.Thread(target=load_data_thread).start()
        logging.info("Load data thread started.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}", exc_info=True)

import spacy
import copy
from .tokenizer import Tokens, Tokenizer

        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        
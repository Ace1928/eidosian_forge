import regex
from .tokenizer import Tokens, Tokenizer
from parlai.utils.logging import logger

        Args:
            annotators: None or empty set (only tokenizes).
            substitutions: if true, normalizes some token types (e.g. quotes).
        
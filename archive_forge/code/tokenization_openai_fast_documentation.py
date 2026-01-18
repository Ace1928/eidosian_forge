from typing import Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_openai import OpenAIGPTTokenizer

    Construct a "fast" GPT Tokenizer (backed by HuggingFace's *tokenizers* library). Based on Byte-Pair-Encoding with
    the following peculiarities:

    - lower case all inputs
    - uses BERT's BasicTokenizer for pre-BPE tokenization

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    
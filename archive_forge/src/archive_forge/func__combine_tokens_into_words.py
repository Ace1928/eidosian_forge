import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
def _combine_tokens_into_words(tokenizer, tokens: List[int], language: str=None, prepend_punctuations: str='"\'“¡¿([{-', append_punctuations: str='"\'.。,，!！?？:：”)]}、'):
    """
    Groups tokens by word. Returns a tuple containing a list of strings with the words, and a list of `token_id`
    sequences with the tokens making up each word.
    """
    if language is None:
        language = tokenizer.language
    if language is None:
        language = 'english'
    if language in {'chinese', 'japanese', 'thai', 'lao', 'myanmar', 'cantonese'}:
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)
    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    return (words, word_tokens, token_indices)
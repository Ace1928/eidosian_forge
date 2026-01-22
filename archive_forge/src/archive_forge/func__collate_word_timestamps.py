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
def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language):
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)
    timings = [{'text': word, 'timestamp': (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1])} for word, indices in zip(words, token_indices)]
    return timings
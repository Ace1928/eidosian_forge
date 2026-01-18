import json
import os
import unicodedata
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
def whitespace_clean(text):
    text = re.sub('\\s+', ' ', text)
    text = text.strip()
    return text
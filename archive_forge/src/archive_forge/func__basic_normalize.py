import json
import os
import re
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np
from tokenizers import AddedToken, pre_tokenizers, processors
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr
def _basic_normalize(self, text, remove_diacritics=False):
    warnings.warn('The private method `_basic_normalize` is deprecated and will be removed in v5 of Transformers.You can normalize an input string using the Whisper basic normalizer using the `basic_normalize` method.')
    return self.basic_normalize(text, remove_diacritics=remove_diacritics)
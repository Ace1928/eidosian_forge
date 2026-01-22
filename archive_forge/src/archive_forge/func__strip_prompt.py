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
@staticmethod
def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
    has_prompt = isinstance(token_ids, list) and token_ids and (token_ids[0] == prompt_token_id)
    if has_prompt:
        if decoder_start_token_id in token_ids:
            return token_ids[token_ids.index(decoder_start_token_id):]
        else:
            return []
    return token_ids
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
def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
    """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
    timestamp_begin = self.all_special_ids[-1] + 1
    outputs = [[]]
    cur_max_timestamp = 0.0
    prev_segments_len = 0.0
    for token in token_ids:
        if token >= timestamp_begin:
            timestamp = float((token - timestamp_begin) * time_precision)
            if timestamp < cur_max_timestamp:
                prev_segments_len += cur_max_timestamp
            cur_max_timestamp = timestamp
            outputs.append(f'<|{timestamp + prev_segments_len:.2f}|>')
            outputs.append([])
        else:
            outputs[-1].append(token)
    outputs = [s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs]
    return ''.join(outputs)
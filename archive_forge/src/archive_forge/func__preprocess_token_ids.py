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
def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool=False):
    """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
    if skip_special_tokens:
        prompt_token_id = self.convert_tokens_to_ids('<|startofprev|>')
        decoder_start_token_id = self.convert_tokens_to_ids('<|startoftranscript|>')
        token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)
    return token_ids
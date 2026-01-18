import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from tokenizers import pre_tokenizers, processors
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils_base import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_markuplm import MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING, MarkupLMTokenizer

        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        
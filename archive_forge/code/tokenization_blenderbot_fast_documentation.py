import json
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers, processors
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_blenderbot import BlenderbotTokenizer

        A very simple chat template that just adds whitespace between messages.
        
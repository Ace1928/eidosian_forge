import json
from typing import Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_gpt2 import GPT2Tokenizer

        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        
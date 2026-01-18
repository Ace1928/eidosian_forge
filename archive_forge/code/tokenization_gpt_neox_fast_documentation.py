import json
from typing import Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        
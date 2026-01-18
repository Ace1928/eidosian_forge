import json
from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers, processors
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_bart import BartTokenizer

        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BART does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        
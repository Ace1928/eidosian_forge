import json
from typing import List, Optional, Tuple
from tokenizers import normalizers
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_mpnet import MPNetTokenizer

        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs

        Returns:
            `List[int]`: List of zeros.
        
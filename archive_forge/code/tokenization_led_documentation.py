import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding, EncodedInput
from ...utils import PaddingStrategy, logging

        Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        
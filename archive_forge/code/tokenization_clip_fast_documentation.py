from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_clip import CLIPTokenizer

        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        
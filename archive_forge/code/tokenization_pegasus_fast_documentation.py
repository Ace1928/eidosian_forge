import os
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

        Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

        - single sequence: `X </s>`
        - pair of sequences: `A B </s>` (not intended use)

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        
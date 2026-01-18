import collections
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A XLMProphetNet sequence has the following format:

        - single sequence: `X [SEP]`
        - pair of sequences: `A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        
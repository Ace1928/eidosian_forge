import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens. A PEGASUS sequence has the following format, where `X` represents the sequence:

        - single sequence: `X </s>`
        - pair of sequences: `A B </s>` (not intended use)

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        
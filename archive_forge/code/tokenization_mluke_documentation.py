import itertools
import json
import os
from collections.abc import Mapping
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging

        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        
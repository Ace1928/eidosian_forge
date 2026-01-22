import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging
from ..xlm_roberta.tokenization_xlm_roberta_fast import (
def _is_valid_text_input(t):
    if isinstance(t, str):
        return True
    elif isinstance(t, (list, tuple)):
        if len(t) == 0:
            return True
        elif isinstance(t[0], str):
            return True
        elif isinstance(t[0], (list, tuple)):
            return len(t[0]) == 0 or isinstance(t[0][0], str)
        else:
            return False
    else:
        return False
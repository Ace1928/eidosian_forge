import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import logging, requires_backends
def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
    """Do not add eos again if user already added it."""
    if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
        warnings.warn(f'This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added.')
        return token_ids
    else:
        return token_ids + [self.eos_token_id]
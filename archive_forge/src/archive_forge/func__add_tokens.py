import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool=False) -> int:
    to_add = []
    for token in new_tokens:
        if isinstance(token, str):
            to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=True, special=special_tokens))
        else:
            to_add.append(token)
    return super()._add_tokens(to_add, special_tokens)
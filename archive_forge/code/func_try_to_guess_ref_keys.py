import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from .base import TaskProcessor
def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
    ref_keys = []
    for name in column_names:
        if 'tag' in name:
            ref_keys.append(name)
    return ref_keys if ref_keys else None
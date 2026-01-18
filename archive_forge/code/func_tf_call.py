import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    if isinstance(examples[0], Mapping):
        examples = [e['input_ids'] for e in examples]
    batch = _tf_collate_batch(examples, self.tokenizer)
    inputs, perm_mask, target_mapping, labels = self.tf_mask_tokens(batch)
    return {'input_ids': inputs, 'perm_mask': perm_mask, 'target_mapping': target_mapping, 'labels': labels}
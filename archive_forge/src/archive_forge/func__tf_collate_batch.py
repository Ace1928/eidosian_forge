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
def _tf_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int]=None):
    import tensorflow as tf
    'Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary.'
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.convert_to_tensor(e, dtype=tf.int64) for e in examples]
    length_of_first = len(examples[0])
    are_tensors_same_length = all((len(x) == length_of_first for x in examples))
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)
    if tokenizer._pad_token is None:
        raise ValueError(f'You are attempting to pad samples but the tokenizer you are using ({tokenizer.__class__.__name__}) does not have a pad token.')
    max_length = max((len(x) for x in examples))
    if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
        max_length = (max_length // pad_to_multiple_of + 1) * pad_to_multiple_of
    result = []
    rank = tf.rank(examples[0])
    paddings = np.zeros((rank, 2), dtype=np.int32)
    for example in examples:
        if tokenizer.padding_side == 'right':
            paddings[0, 1] = max_length - len(example)
        else:
            paddings[0, 0] = max_length - len(example)
        result.append(tf.pad(example, paddings, constant_values=tokenizer.pad_token_id))
    return tf.stack(result, axis=0)
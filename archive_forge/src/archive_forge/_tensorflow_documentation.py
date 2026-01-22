import math
import re
from typing import TYPE_CHECKING, Dict
from ._base import MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory

    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    Taken from https://github.com/huggingface/transformers/blob/74d9d0cebb0263a3f8ab9c280569170cc74651d0/src/transformers/modeling_tf_utils.py#L608.
    NOTE: why not `tensor.numpy().nbytes`?
    Example:
    ```py
    >>> _dtype_byte_size(tf.float32)
    4
    ```
    
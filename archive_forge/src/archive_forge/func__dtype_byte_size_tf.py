import math
import re
from typing import TYPE_CHECKING, Dict
from ._base import MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory
def _dtype_byte_size_tf(dtype) -> float:
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    Taken from https://github.com/huggingface/transformers/blob/74d9d0cebb0263a3f8ab9c280569170cc74651d0/src/transformers/modeling_tf_utils.py#L608.
    NOTE: why not `tensor.numpy().nbytes`?
    Example:
    ```py
    >>> _dtype_byte_size(tf.float32)
    4
    ```
    """
    import tensorflow as tf
    if dtype == tf.bool:
        return 1 / 8
    bit_search = re.search('[^\\d](\\d+)$', dtype.name)
    if bit_search is None:
        raise ValueError(f'`dtype` is not a valid dtype: {dtype}.')
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8
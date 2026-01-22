from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
@Deprecated(help='RLlib itself has no use for this anymore.', error=False)
def aligned_array(size: int, dtype, align: int=64) -> np.ndarray:
    """Returns an array of a given size that is 64-byte aligned.

    The returned array can be efficiently copied into GPU memory by TensorFlow.

    Args:
        size: The size (total number of items) of the array. For example,
            array([[0.0, 1.0], [2.0, 3.0]]) would have size=4.
        dtype: The numpy dtype of the array.
        align: The alignment to use.

    Returns:
        A np.ndarray with the given specifications.
    """
    n = size * dtype.itemsize
    empty = np.empty(n + (align - 1), dtype=np.uint8)
    data_align = empty.ctypes.data % align
    offset = 0 if data_align == 0 else align - data_align
    if n == 0:
        output = empty[offset:offset + 1][0:0].view(dtype)
    else:
        output = empty[offset:offset + n].view(dtype)
    assert len(output) == size, len(output)
    assert output.ctypes.data % align == 0, output.ctypes.data
    return output
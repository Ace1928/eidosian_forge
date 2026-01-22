import logging
import threading
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import TensorStructType
from ray.rllib.utils.serialization import _serialize_ndarray, _deserialize_ndarray
from ray.rllib.utils.deprecation import deprecation_warning
def _helper(x, rs, buffer, shape):
    if shape is None:
        return x
    orig_dtype = x.dtype
    if update:
        if len(x.shape) == len(rs.shape) + 1:
            for i in range(x.shape[0]):
                rs.push(x[i])
                buffer.push(x[i])
        else:
            rs.push(x)
            buffer.push(x)
    if self.demean:
        x = x - rs.mean
    if self.destd:
        x = x / (rs.std + SMALL_NUMBER)
    if self.clip:
        x = np.clip(x, -self.clip, self.clip)
    return x.astype(orig_dtype)
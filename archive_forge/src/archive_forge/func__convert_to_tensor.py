import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _convert_to_tensor(value, name=None, preferred_dtype=None):
    """Converts to tensor avoiding an eager bug that loses float precision."""
    if context.executing_eagerly() and preferred_dtype is not None and (preferred_dtype.is_integer or preferred_dtype.is_bool):
        v = ops.convert_to_tensor(value, name=name)
        if v.dtype.is_floating:
            return v
    return ops.convert_to_tensor(value, name=name, preferred_dtype=preferred_dtype)
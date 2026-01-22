import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _maybe_get_dtype(x):
    """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
    if isinstance(x, numbers.Real):
        return x
    if isinstance(x, indexed_slices.IndexedSlices) or tensor_util.is_tf_type(x):
        return _to_numpy_type(x.dtype)
    if isinstance(x, dtypes.DType):
        return x.as_numpy_dtype
    if isinstance(x, (list, tuple)):
        raise ValueError(f'Cannot find dtype for type inference from argument `x` of a sequence type {type(x)}. For sequences, please call this function on each element individually.')
    return x
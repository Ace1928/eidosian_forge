import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _compute_size_of_strided_dim(shrink, spec, size):
    """Computes the size of a single strided slice dimension."""
    unknown = None
    use_full_range = None
    if shrink:
        return 1
    if size is unknown or size.value is unknown:
        return unknown
    size = size.value
    stride = spec.step
    if stride is not unknown:
        if stride == 0:
            return unknown
        stride = spec.step
        valid_range = [0, size] if stride > 0 else [-1, size - 1]

        def canonical(x, c):
            if x is use_full_range:
                return valid_range[c] if stride > 0 else valid_range[c + 1 & 1]
            else:
                x_fwd = size + x if x < 0 else x
                return max(valid_range[0], min(valid_range[1], x_fwd))
        begin = canonical(spec.start, 0)
        end = canonical(spec.stop, 1)
        interval_length = end - begin
        if interval_length == 0 or (interval_length < 0) != (stride < 0):
            return 0
        else:
            remainder = 1 if interval_length % stride != 0 else 0
            return interval_length // stride + remainder
    else:
        return unknown
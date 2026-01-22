import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_linalg_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _use_composite_impl(fast, tensor_shape):
    """Determines whether to use the composite or specialized CPU kernel.

    When the total size of the tensor is larger than the cache size and the
    batch size is large compared to the smallest matrix dimension, then the
    composite implementation is inefficient since it has to read the entire
    tensor from memory multiple times. In this case we fall back to the
    original CPU kernel, which does all the computational steps on each
    matrix separately.

    Only fast mode is supported by the composite impl, so `False` is returned
    if `fast` is `False`.

    Args:
      fast: bool indicating if fast mode in the solver was requested.
      tensor_shape: The shape of the tensor.

    Returns:
      True if the composite impl should be used. False otherwise.
    """
    if fast is False:
        return False
    batch_shape = tensor_shape[:-2]
    matrix_shape = tensor_shape[-2:]
    if not tensor_shape.is_fully_defined():
        return True
    tensor_size = tensor_shape.num_elements() * matrix.dtype.size
    is_io_bound = batch_shape.num_elements() > np.min(matrix_shape)
    L2_CACHE_SIZE_GUESSTIMATE = 256000
    if tensor_size > L2_CACHE_SIZE_GUESSTIMATE and is_io_bound:
        return False
    else:
        return True
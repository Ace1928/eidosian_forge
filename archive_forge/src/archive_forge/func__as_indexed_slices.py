import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@dispatch.add_dispatch_support
def _as_indexed_slices(x, optimize=True):
    """Convert 'x' to IndexedSlices.

  Convert a dense Tensor to a block-sparse IndexedSlices.

  Args:
    x: Either a Tensor object, or an IndexedSlices object.
    optimize: if true, attempt to optimize the conversion of 'x'.

  Returns:
    An IndexedSlices object.

  Raises:
    TypeError: If 'x' is not a Tensor or an IndexedSlices object.
  """
    if not isinstance(x, (tensor_lib.Tensor, indexed_slices.IndexedSlices)):
        raise TypeError(f'Not a Tensor or IndexedSlices: {type(x)}.')
    if isinstance(x, indexed_slices.IndexedSlices):
        return x
    x_shape = array_ops.shape_internal(x, optimize=optimize)
    return indexed_slices.IndexedSlices(x, range(0, x_shape[0]), x_shape)
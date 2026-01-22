import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.array', v1=[])
@np_utils.np_doc_only('array')
def array(val, dtype=None, copy=True, ndmin=0):
    """Since Tensors are immutable, a copy is made only if val is placed on a

  different device than the current one. Even if `copy` is False, a new Tensor
  may need to be built to satisfy `dtype` and `ndim`. This is used only if `val`
  is an ndarray or a Tensor.
  """
    if dtype:
        dtype = np_utils.result_type(dtype)
    return _array_internal(val, dtype, copy, ndmin)
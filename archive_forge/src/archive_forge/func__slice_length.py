from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _slice_length(value_length, slice_key):
    """Computes the number of elements in a slice of a value with a given length.

  Returns the equivalent of: `len(range(value_length)[slice_key])`

  Args:
    value_length: Scalar int `Tensor`: the length of the value being sliced.
    slice_key: A `slice` object used to slice elements from the value.

  Returns:
    The number of elements in the sliced value.
  """
    zeros = array_ops.zeros(value_length, dtype=dtypes.bool)
    return array_ops.size(zeros[slice_key], out_type=value_length.dtype)
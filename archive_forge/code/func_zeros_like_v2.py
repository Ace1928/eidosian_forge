from typing import Sequence
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_types(array_ops.zeros_like_v2, StructuredTensor)
def zeros_like_v2(input, dtype=None, name=None, layout=None):
    """Replace every object with a zero.

  Example:
  >>> st = StructuredTensor.from_pyval([{"x":[3]}, {"x":[4,5]}])
  >>> tf.zeros_like(st)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0.0, 0.0], dtype=float32)>
  >>> st = StructuredTensor.from_pyval([[{"x":[3]}], [{"x":[4,5]}, {"x":[]}]])
  >>> tf.zeros_like(st, dtype=tf.int32)
  <tf.RaggedTensor [[0], [0, 0]]>

  Args:
    input: a structured tensor.
    dtype: the dtype of the resulting zeros. (default is tf.float32)
    name: a name for the op.
    layout: Optional Layout. Only supports replicated layout.

  Returns:
    a tensor of zeros of the same shape.
  """
    if layout is not None and (not layout.is_fully_replicated()):
        raise ValueError(f'StructuredTensor only allows replicated layout. got {layout}')
    if dtype is None:
        dtype = dtypes.float32
    with ops.name_scope(name, 'zeros_like', [input]) as name:
        if not input.row_partitions:
            if input.nrows() is not None:
                return array_ops.zeros([input.nrows()], dtype, layout=layout)
            else:
                return array_ops.zeros([], dtype, layout=layout)
        last_row_partition = input.row_partitions[-1]
        result = ragged_tensor.RaggedTensor._from_nested_row_partitions(array_ops.zeros(last_row_partition.nvals(), dtype=dtype), input.row_partitions)
        return result
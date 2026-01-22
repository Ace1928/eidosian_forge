from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
def _increase_rank_to(t, rank):
    """Adds *trailing* size-1 dimensions to `t` until it has the given rank."""
    if isinstance(t, ragged_tensor.RaggedTensor):
        return t.with_values(_increase_rank_to(t, rank - 1))
    else:
        old_dims = array_ops.shape(t)
        new_dims = array_ops.ones([rank - array_ops.rank(t)], old_dims.dtype)
        new_shape = array_ops.concat([old_dims, new_dims], axis=0)
        return array_ops.reshape(t, new_shape)
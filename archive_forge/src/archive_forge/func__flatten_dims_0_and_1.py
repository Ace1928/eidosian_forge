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
def _flatten_dims_0_and_1(t):
    """Returns a copy of `t` with the outer two dimensions merged."""
    if isinstance(t, ragged_tensor.RaggedTensor):
        return t.values
    else:
        t_shape = array_ops.shape(t)
        return array_ops.reshape(t, array_ops.concat([[-1], t_shape[2:]], axis=0))
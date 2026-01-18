from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
def to_tensor(rt_input, default_value=None, name=None):
    if ragged_tensor.is_ragged(rt_input):
        return rt_input.to_tensor(default_value, name)
    else:
        return rt_input
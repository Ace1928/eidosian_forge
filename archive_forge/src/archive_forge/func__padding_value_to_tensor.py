import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops
def _padding_value_to_tensor(value, output_type):
    """Converts the padding value to a tensor.

  Args:
    value: The padding value.
    output_type: Its expected dtype.

  Returns:
    A scalar `Tensor`.

  Raises:
    ValueError: if the padding value is not a scalar.
    TypeError: if the padding value's type does not match `output_type`.
  """
    value = ops.convert_to_tensor(value, name='padding_value')
    if not value.shape.is_compatible_with(tensor_shape.TensorShape([])):
        raise ValueError(f'Invalid `padding_values`. `padding_values` values should be scalars, but got {value.shape}.')
    if value.dtype != output_type:
        raise TypeError(f'Invalid `padding_values`. `padding_values` values type {value.dtype} does not match type {output_type} of the corresponding input component.')
    return value
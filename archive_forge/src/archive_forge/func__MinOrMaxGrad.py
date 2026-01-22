import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
def _MinOrMaxGrad(op, grad):
    """Gradient for Min or Max. Amazingly it's precisely the same code."""
    input_shape = array_ops.shape(op.inputs[0])
    y = op.outputs[0]
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        y = array_ops.reshape(y, output_shape_kept_dims)
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    else:
        output_shape_kept_dims = array_ops.shape(y)
    indicators = math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype)
    num_selected = array_ops.reshape(math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims)
    return [math_ops.divide(indicators, num_selected) * grad, None]
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
@ops.RegisterGradient('SelectV2')
def _SelectGradV2(op, grad):
    c = op.inputs[0]
    x = op.inputs[1]
    y = op.inputs[2]
    zeros = array_ops.zeros([], dtype=grad.dtype.base_dtype)
    gx = array_ops.where_v2(c, grad, zeros)
    x_shape = array_ops.shape(x)
    output_shape = array_ops.shape(op.outputs[0])
    reduce_x, _ = gen_array_ops.broadcast_gradient_args(x_shape, output_shape)
    gx = math_ops.reduce_sum(gx, keepdims=True, axis=reduce_x)
    gx = array_ops.reshape(gx, x_shape)
    gy = array_ops.where_v2(c, zeros, grad)
    y_shape = array_ops.shape(y)
    reduce_y, _ = gen_array_ops.broadcast_gradient_args(y_shape, output_shape)
    gy = math_ops.reduce_sum(gy, keepdims=True, axis=reduce_y)
    gy = array_ops.reshape(gy, y_shape)
    return (None, gx, gy)
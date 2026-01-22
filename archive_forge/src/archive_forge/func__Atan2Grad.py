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
@ops.RegisterGradient('Atan2')
def _Atan2Grad(op, grad):
    """Returns grad * x / (x^2 + y^2), grad * -y / (x^2 + y^2)."""
    y = op.inputs[0]
    x = op.inputs[1]
    with ops.control_dependencies([grad]):
        (sx, rx, must_reduce_x), (sy, ry, must_reduce_y) = SmartBroadcastGradientArgs(x, y, grad)
        grad_inv = grad / (math_ops.square(x) + math_ops.square(y))
        gx = -y * grad_inv
        if must_reduce_x:
            gx = array_ops.reshape(math_ops.reduce_sum(gx, rx), sx)
        gy = x * grad_inv
        if must_reduce_y:
            gy = array_ops.reshape(math_ops.reduce_sum(gy, ry), sy)
        return (gy, gx)
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
@ops.RegisterGradient('DivNoNan')
def _DivNoNanGrad(op, grad):
    """DivNoNan op gradient."""
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    x = math_ops.conj(x)
    y = math_ops.conj(y)
    return (array_ops.reshape(math_ops.reduce_sum(math_ops.div_no_nan(grad, y), rx), sx), array_ops.reshape(math_ops.reduce_sum(grad * math_ops.div_no_nan(math_ops.div_no_nan(-x, y), y), ry), sy))
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
@ops.RegisterGradient('MulNoNan')
def _MulNoNanGrad(op, grad):
    """The gradient of scalar multiplication with NaN-suppression."""
    x = op.inputs[0]
    y = op.inputs[1]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (gen_math_ops.mul_no_nan(grad, y), gen_math_ops.mul_no_nan(x, grad))
    assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, ' vs. ', y.dtype)
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    return (array_ops.reshape(math_ops.reduce_sum(gen_math_ops.mul_no_nan(grad, y), rx), sx), array_ops.reshape(math_ops.reduce_sum(gen_math_ops.mul_no_nan(x, grad), ry), sy))
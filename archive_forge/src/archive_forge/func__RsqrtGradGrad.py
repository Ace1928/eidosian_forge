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
@ops.RegisterGradient('RsqrtGrad')
def _RsqrtGradGrad(op, grad):
    """Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3."""
    a = op.inputs[0]
    b = op.inputs[1]
    with ops.control_dependencies([grad]):
        ca = math_ops.conj(a)
        cg = math_ops.conj(grad)
        grad_a = -1.5 * cg * b * math_ops.square(ca)
        grad_b = gen_math_ops.rsqrt_grad(ca, grad)
        return (grad_a, grad_b)
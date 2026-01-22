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
@ops.RegisterGradient('Angle')
def _AngleGrad(op, grad):
    """Returns -grad / (Im(x) + iRe(x))"""
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        re = math_ops.real(x)
        im = math_ops.imag(x)
        z = math_ops.reciprocal(math_ops.complex(im, re))
        zero = constant_op.constant(0, dtype=grad.dtype)
        complex_grad = math_ops.complex(grad, zero)
        return -complex_grad * z
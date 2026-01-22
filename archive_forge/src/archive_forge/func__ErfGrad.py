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
@ops.RegisterGradient('Erf')
def _ErfGrad(op, grad):
    """Returns grad * 2/sqrt(pi) * exp(-x**2)."""
    x = op.inputs[0]
    two_over_root_pi = constant_op.constant(2 / np.sqrt(np.pi), dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * two_over_root_pi * math_ops.exp(-math_ops.square(x))
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
def _MatMulGradAgainstSecondOnly(op, grad):
    """Gradient for MatMul, only for the second input."""
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    a = math_ops.conj(op.inputs[0])
    if not t_a and (not t_b):
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True, grad_b=True)
    elif not t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, grad_b=True)
    elif t_a and (not t_b):
        grad_b = gen_math_ops.mat_mul(a, grad, grad_b=True)
    elif t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True, grad_b=True)
    return (None, grad_b)
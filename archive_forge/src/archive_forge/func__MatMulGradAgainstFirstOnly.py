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
def _MatMulGradAgainstFirstOnly(op, grad):
    """Gradient for MatMul, only for the first input."""
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    b = math_ops.conj(op.inputs[1])
    if not t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True, grad_a=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b, grad_a=True)
    elif t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True, grad_a=True)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True, grad_a=True)
    return (grad_a, None)
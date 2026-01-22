from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('MatrixSolve')
def _MatrixSolveGrad(op, grad):
    """Gradient for MatrixSolve."""
    a = op.inputs[0]
    adjoint_a = op.get_attr('adjoint')
    c = op.outputs[0]
    grad_b = linalg_ops.matrix_solve(a, grad, adjoint=not adjoint_a)
    if adjoint_a:
        grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
    else:
        grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
    return (grad_a, grad_b)
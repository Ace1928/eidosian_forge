from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('MatrixDeterminant')
def _MatrixDeterminantGrad(op, grad):
    """Gradient for MatrixDeterminant."""
    a = op.inputs[0]
    c = op.outputs[0]
    a_adj_inv = linalg_ops.matrix_inverse(a, adjoint=True)
    multipliers = array_ops.reshape(grad * c, array_ops.concat([array_ops.shape(c), [1, 1]], 0))
    return multipliers * a_adj_inv
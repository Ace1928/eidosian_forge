from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('SelfAdjointEigV2')
def _SelfAdjointEigV2Grad(op, grad_e, grad_v):
    """Gradient for SelfAdjointEigV2."""
    e = op.outputs[0]
    compute_v = op.get_attr('compute_v')
    with ops.control_dependencies([grad_e, grad_v]):
        if compute_v:
            v = op.outputs[1]
            f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(e, -2) - array_ops.expand_dims(e, -1)), array_ops.zeros_like(e))
            grad_a = math_ops.matmul(v, math_ops.matmul(array_ops.matrix_diag(grad_e) + f * math_ops.matmul(v, grad_v, adjoint_a=True), v, adjoint_b=True))
        else:
            _, v = linalg_ops.self_adjoint_eig(op.inputs[0])
            grad_a = math_ops.matmul(v, math_ops.matmul(array_ops.matrix_diag(grad_e), v, adjoint_b=True))
        grad_a = array_ops.matrix_band_part(grad_a + _linalg.adjoint(grad_a), -1, 0)
        grad_a = array_ops.matrix_set_diag(grad_a, 0.5 * array_ops.matrix_diag_part(grad_a))
        return grad_a
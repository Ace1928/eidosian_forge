from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('Svd')
def _SvdGrad(op, grad_s, grad_u, grad_v):
    """Gradient for the singular value decomposition."""
    a = op.inputs[0]
    a_shape = a.get_shape().with_rank_at_least(2)
    grad_s = math_ops.cast(grad_s, a.dtype)
    grad_s_mat = array_ops.matrix_diag(grad_s)
    if not op.get_attr('compute_uv'):
        s, u, v = linalg_ops.svd(a, compute_uv=True)
        grad_a = math_ops.matmul(u, math_ops.matmul(grad_s_mat, v, adjoint_b=True))
        grad_a.set_shape(a_shape)
        return grad_a
    full_matrices = op.get_attr('full_matrices')
    grad_u_shape = grad_u.get_shape().with_rank_at_least(2)
    grad_v_shape = grad_v.get_shape().with_rank_at_least(2)
    m = a_shape.dims[-2].merge_with(grad_u_shape[-2])
    n = a_shape.dims[-1].merge_with(grad_v_shape[-2])
    batch_shape = a_shape[:-2].merge_with(grad_u_shape[:-2]).merge_with(grad_v_shape[:-2])
    a_shape = batch_shape.concatenate([m, n])
    m = a_shape.dims[-2].value
    n = a_shape.dims[-1].value
    if m is None or n is None:
        raise NotImplementedError('SVD gradient has not been implemented for input with unknown inner matrix shape.')
    s = op.outputs[0]
    u = op.outputs[1]
    v = op.outputs[2]
    s = math_ops.cast(s, a.dtype)
    use_adjoint = False
    if m > n:
        use_adjoint = True
        m, n = (n, m)
        u, v = (v, u)
        grad_u, grad_v = (grad_v, grad_u)
    with ops.control_dependencies([grad_s, grad_u, grad_v]):
        if full_matrices and abs(m - n) > 1:
            raise NotImplementedError(f'svd gradient is not implemented for abs(m - n) > 1 when full_matrices is True. Received: m={m} and n={n} from op input={a} with shape={a_shape}.')
        s_mat = array_ops.matrix_diag(s)
        s2 = math_ops.square(s)
        s_shape = array_ops.shape(s)
        f = array_ops.matrix_set_diag(_SafeReciprocal(array_ops.expand_dims(s2, -2) - array_ops.expand_dims(s2, -1)), array_ops.zeros_like(s))
        s_inv_mat = array_ops.matrix_diag(_SafeReciprocal(s))
        v1 = v[..., :, :m]
        grad_v1 = grad_v[..., :, :m]
        u_gu = math_ops.matmul(u, grad_u, adjoint_a=True)
        v_gv = math_ops.matmul(v1, grad_v1, adjoint_a=True)
        f_u = f * u_gu
        f_v = f * v_gv
        term1_nouv = grad_s_mat + math_ops.matmul(f_u + _linalg.adjoint(f_u), s_mat) + math_ops.matmul(s_mat, f_v + _linalg.adjoint(f_v))
        term1 = math_ops.matmul(u, math_ops.matmul(term1_nouv, v1, adjoint_b=True))
        if m == n:
            grad_a_before_transpose = term1
        else:
            gv1t = array_ops.matrix_transpose(grad_v1, conjugate=True)
            gv1t_v1 = math_ops.matmul(gv1t, v1)
            term2_nous = gv1t - math_ops.matmul(gv1t_v1, v1, adjoint_b=True)
            if full_matrices:
                v2 = v[..., :, m:n]
                grad_v2 = grad_v[..., :, m:n]
                v1t_gv2 = math_ops.matmul(v1, grad_v2, adjoint_a=True)
                term2_nous -= math_ops.matmul(v1t_gv2, v2, adjoint_b=True)
            u_s_inv = math_ops.matmul(u, s_inv_mat)
            term2 = math_ops.matmul(u_s_inv, term2_nous)
            grad_a_before_transpose = term1 + term2
        if a.dtype.is_complex:
            eye = _linalg.eye(s_shape[-1], batch_shape=s_shape[:-1], dtype=a.dtype)
            l = eye * v_gv
            term3_nouv = math_ops.matmul(s_inv_mat, _linalg.adjoint(l) - l)
            term3 = 1 / 2.0 * math_ops.matmul(u, math_ops.matmul(term3_nouv, v1, adjoint_b=True))
            grad_a_before_transpose += term3
        if use_adjoint:
            grad_a = array_ops.matrix_transpose(grad_a_before_transpose, conjugate=True)
        else:
            grad_a = grad_a_before_transpose
        grad_a.set_shape(a_shape)
        return grad_a
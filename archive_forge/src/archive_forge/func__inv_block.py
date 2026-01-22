from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_block(M, iszerofunc=_iszero):
    """Calculates the inverse using BLOCKWISE inversion.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    i = M.shape[0]
    if i <= 20:
        return M.inv(method='LU', iszerofunc=_iszero)
    A = M[:i // 2, :i // 2]
    B = M[:i // 2, i // 2:]
    C = M[i // 2:, :i // 2]
    D = M[i // 2:, i // 2:]
    try:
        D_inv = _inv_block(D)
    except NonInvertibleMatrixError:
        return M.inv(method='LU', iszerofunc=_iszero)
    B_D_i = B * D_inv
    BDC = B_D_i * C
    A_n = A - BDC
    try:
        A_n = _inv_block(A_n)
    except NonInvertibleMatrixError:
        return M.inv(method='LU', iszerofunc=_iszero)
    B_n = -A_n * B_D_i
    dc = D_inv * C
    C_n = -dc * A_n
    D_n = D_inv + dc * -B_n
    nn = BlockMatrix([[A_n, B_n], [C_n, D_n]]).as_explicit()
    return nn
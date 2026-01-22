from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _pinv_rank_decomposition(M):
    """Subroutine for rank decomposition

    With rank decompositions, `A` can be decomposed into two full-
    rank matrices, and each matrix can take pseudoinverse
    individually.
    """
    if M.is_zero_matrix:
        return M.H
    B, C = M.rank_decomposition()
    Bp = _pinv_full_rank(B)
    Cp = _pinv_full_rank(C)
    return Cp.multiply(Bp)
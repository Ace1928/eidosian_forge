from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _pinv_diagonalization(M):
    """Subroutine using diagonalization

    This routine can sometimes fail if SymPy's eigenvalue
    computation is not reliable.
    """
    if M.is_zero_matrix:
        return M.H
    A = M
    AH = M.H
    try:
        if M.rows >= M.cols:
            P, D = AH.multiply(A).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)
            return P.multiply(D_pinv).multiply(P.H).multiply(AH)
        else:
            P, D = A.multiply(AH).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)
            return AH.multiply(P).multiply(D_pinv).multiply(P.H)
    except MatrixError:
        raise NotImplementedError('pinv for rank-deficient matrices where diagonalization of A.H*A fails is not supported yet.')
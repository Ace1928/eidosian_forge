from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_LDL(M, iszerofunc=_iszero):
    """Calculates the inverse using LDL decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    """
    _verify_invertible(M, iszerofunc=iszerofunc)
    return M.LDLsolve(M.eye(M.rows))
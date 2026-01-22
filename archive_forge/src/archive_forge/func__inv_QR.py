from sympy.core.numbers import mod_inverse
from .common import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero
def _inv_QR(M, iszerofunc=_iszero):
    """Calculates the inverse using QR decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
    _verify_invertible(M, iszerofunc=iszerofunc)
    return M.QRsolve(M.eye(M.rows))
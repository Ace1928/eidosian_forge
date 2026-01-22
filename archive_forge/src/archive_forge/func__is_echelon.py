from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""
    if M.rows <= 0 or M.cols <= 0:
        return True
    zeros_below = all((iszerofunc(t) for t in M[1:, 0]))
    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)
    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)
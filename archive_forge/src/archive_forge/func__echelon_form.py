from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """
    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify
    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc, normalize_last=True, normalize=False, zero_above=False)
    if with_pivots:
        return (mat, pivots)
    return mat
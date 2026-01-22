from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols
from .common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero
def _solve_least_squares(M, rhs, method='CH'):
    """Return the least-square fit to the data.

    Parameters
    ==========

    rhs : Matrix
        Vector representing the right hand side of the linear equation.

    method : string or boolean, optional
        If set to ``'CH'``, ``cholesky_solve`` routine will be used.

        If set to ``'LDL'``, ``LDLsolve`` routine will be used.

        If set to ``'QR'``, ``QRsolve`` routine will be used.

        If set to ``'PINV'``, ``pinv_solve`` routine will be used.

        Otherwise, the conjugate of ``M`` will be used to create a system
        of equations that is passed to ``solve`` along with the hint
        defined by ``method``.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution.

    Examples
    ========

    >>> from sympy import Matrix, ones
    >>> A = Matrix([1, 2, 3])
    >>> B = Matrix([2, 3, 4])
    >>> S = Matrix(A.row_join(B))
    >>> S
    Matrix([
    [1, 2],
    [2, 3],
    [3, 4]])

    If each line of S represent coefficients of Ax + By
    and x and y are [2, 3] then S*xy is:

    >>> r = S*Matrix([2, 3]); r
    Matrix([
    [ 8],
    [13],
    [18]])

    But let's add 1 to the middle value and then solve for the
    least-squares value of xy:

    >>> xy = S.solve_least_squares(Matrix([8, 14, 18])); xy
    Matrix([
    [ 5/3],
    [10/3]])

    The error is given by S*xy - r:

    >>> S*xy - r
    Matrix([
    [1/3],
    [1/3],
    [1/3]])
    >>> _.norm().n(2)
    0.58

    If a different xy is used, the norm will be higher:

    >>> xy += ones(2, 1)/10
    >>> (S*xy - r).norm().n(2)
    1.5

    """
    if method == 'CH':
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        return M.QRsolve(rhs)
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    else:
        t = M.H
        return (t * M).solve(t * rhs, method=method)
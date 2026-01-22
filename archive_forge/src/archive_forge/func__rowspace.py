from .utilities import _iszero
def _rowspace(M, simplify=False):
    """Returns a list of vectors that span the row space of ``M``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.rowspace()
    [Matrix([[1, 3, 0]]), Matrix([[0, 0, 6]])]
    """
    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)
    return [reduced.row(i) for i in range(len(pivots))]
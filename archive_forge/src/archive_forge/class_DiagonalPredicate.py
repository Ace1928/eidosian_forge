from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class DiagonalPredicate(Predicate):
    """
    Diagonal matrix predicate.

    Explanation
    ===========

    ``Q.diagonal(x)`` is true iff ``x`` is a diagonal matrix. A diagonal
    matrix is a matrix in which the entries outside the main diagonal
    are all zero.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, ZeroMatrix
    >>> X = MatrixSymbol('X', 2, 2)
    >>> ask(Q.diagonal(ZeroMatrix(3, 3)))
    True
    >>> ask(Q.diagonal(X), Q.lower_triangular(X) &
    ...     Q.upper_triangular(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Diagonal_matrix

    """
    name = 'diagonal'
    handler = Dispatcher('DiagonalHandler', doc="Handler for key 'diagonal'.")
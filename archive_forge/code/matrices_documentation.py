from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher

    Unit triangular matrix predicate.

    Explanation
    ===========

    A unit triangular matrix is a triangular matrix with 1s
    on the diagonal.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.triangular(X), Q.unit_triangular(X))
    True

    
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class OrthogonalPredicate(Predicate):
    """
    Orthogonal matrix predicate.

    Explanation
    ===========

    ``Q.orthogonal(x)`` is true iff ``x`` is an orthogonal matrix.
    A square matrix ``M`` is an orthogonal matrix if it satisfies
    ``M^TM = MM^T = I`` where ``M^T`` is the transpose matrix of
    ``M`` and ``I`` is an identity matrix. Note that an orthogonal
    matrix is necessarily invertible.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol, Identity
    >>> X = MatrixSymbol('X', 2, 2)
    >>> Y = MatrixSymbol('Y', 2, 3)
    >>> Z = MatrixSymbol('Z', 2, 2)
    >>> ask(Q.orthogonal(Y))
    False
    >>> ask(Q.orthogonal(X*Z*X), Q.orthogonal(X) & Q.orthogonal(Z))
    True
    >>> ask(Q.orthogonal(Identity(3)))
    True
    >>> ask(Q.invertible(X), Q.orthogonal(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Orthogonal_matrix

    """
    name = 'orthogonal'
    handler = Dispatcher('OrthogonalHandler', doc="Handler for key 'orthogonal'.")
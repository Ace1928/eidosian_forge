from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class NormalPredicate(Predicate):
    """
    Normal matrix predicate.

    A matrix is normal if it commutes with its conjugate transpose.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.normal(X), Q.unitary(X))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_matrix

    """
    name = 'normal'
    handler = Dispatcher('NormalHandler', doc="Predicate fore key 'normal'.")
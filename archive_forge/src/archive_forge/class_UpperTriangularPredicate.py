from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class UpperTriangularPredicate(Predicate):
    """
    Upper triangular matrix predicate.

    Explanation
    ===========

    A matrix $M$ is called upper triangular matrix if :math:`M_{ij}=0`
    for :math:`i<j`.

    Examples
    ========

    >>> from sympy import Q, ask, ZeroMatrix, Identity
    >>> ask(Q.upper_triangular(Identity(3)))
    True
    >>> ask(Q.upper_triangular(ZeroMatrix(3, 3)))
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UpperTriangularMatrix.html

    """
    name = 'upper_triangular'
    handler = Dispatcher('UpperTriangularHandler', doc="Handler for key 'upper_triangular'.")
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class IntegerElementsPredicate(Predicate):
    """
    Integer elements matrix predicate.

    Explanation
    ===========

    ``Q.integer_elements(x)`` is true iff all the elements of ``x``
    are integers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.integer(X[1, 2]), Q.integer_elements(X))
    True

    """
    name = 'integer_elements'
    handler = Dispatcher('IntegerElementsHandler', doc="Handler for key 'integer_elements'.")
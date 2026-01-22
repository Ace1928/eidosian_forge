from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ComplexElementsPredicate(Predicate):
    """
    Complex elements matrix predicate.

    Explanation
    ===========

    ``Q.complex_elements(x)`` is true iff all the elements of ``x``
    are complex numbers.

    Examples
    ========

    >>> from sympy import Q, ask, MatrixSymbol
    >>> X = MatrixSymbol('X', 4, 4)
    >>> ask(Q.complex(X[1, 2]), Q.complex_elements(X))
    True
    >>> ask(Q.complex_elements(X), Q.integer_elements(X))
    True

    """
    name = 'complex_elements'
    handler = Dispatcher('ComplexElementsHandler', doc="Handler for key 'complex_elements'.")
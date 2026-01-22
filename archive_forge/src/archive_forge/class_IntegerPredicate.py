from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'
    handler = Dispatcher('IntegerHandler', doc='Handler for Q.integer.\n\nTest that an expression belongs to the field of integer numbers.')
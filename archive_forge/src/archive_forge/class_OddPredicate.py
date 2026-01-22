from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class OddPredicate(Predicate):
    """
    Odd number predicate.

    Explanation
    ===========

    ``ask(Q.odd(x))`` is true iff ``x`` belongs to the set of odd numbers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.odd(0))
    False
    >>> ask(Q.odd(2))
    False
    >>> ask(Q.odd(3))
    True
    >>> ask(Q.odd(pi))
    False

    """
    name = 'odd'
    handler = Dispatcher('OddHandler', doc="Handler for key 'odd'. Test that an expression represents an odd number.")
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class EvenPredicate(Predicate):
    """
    Even number predicate.

    Explanation
    ===========

    ``ask(Q.even(x))`` is true iff ``x`` belongs to the set of even
    integers.

    Examples
    ========

    >>> from sympy import Q, ask, pi
    >>> ask(Q.even(0))
    True
    >>> ask(Q.even(2))
    True
    >>> ask(Q.even(3))
    False
    >>> ask(Q.even(pi))
    False

    """
    name = 'even'
    handler = Dispatcher('EvenHandler', doc="Handler for key 'even'.")
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class NegativeInfinitePredicate(Predicate):
    """
    Negative infinity predicate.

    ``Q.negative_infinite(x)`` is true iff ``x`` is negative infinity ``-oo``.
    """
    name = 'negative_infinite'
    handler = Dispatcher('NegativeInfiniteHandler')
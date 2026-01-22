from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ExtendedNonNegativePredicate(Predicate):
    """
    Nonnegative extended real number predicate.

    Explanation
    ===========

    ``ask(Q.extended_nonnegative(x))`` is true iff ``x`` is extended real and
    ``x`` is not negative.

    Examples
    ========

    >>> from sympy import ask, I, oo, Q
    >>> ask(Q.extended_nonnegative(-1))
    False
    >>> ask(Q.extended_nonnegative(oo))
    True
    >>> ask(Q.extended_nonnegative(0))
    True
    >>> ask(Q.extended_nonnegative(I))
    False

    """
    name = 'extended_nonnegative'
    handler = Dispatcher('ExtendedNonNegativeHandler')
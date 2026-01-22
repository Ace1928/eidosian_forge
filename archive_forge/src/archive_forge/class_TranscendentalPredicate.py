from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class TranscendentalPredicate(Predicate):
    """
    Transcedental number predicate.

    Explanation
    ===========

    ``Q.transcendental(x)`` is true iff ``x`` belongs to the set of
    transcendental numbers. A transcendental number is a real
    or complex number that is not algebraic.

    """
    name = 'transcendental'
    handler = Dispatcher('Transcendental', doc='Handler for Q.transcendental key.')
from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
class ReversedGradedLexOrder(MonomialOrder):
    """Reversed graded lexicographic order of monomials. """
    alias = 'grevlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), tuple(reversed([-m for m in monomial])))
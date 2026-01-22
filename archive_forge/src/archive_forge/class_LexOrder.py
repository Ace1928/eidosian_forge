from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
class LexOrder(MonomialOrder):
    """Lexicographic order of monomials. """
    alias = 'lex'
    is_global = True
    is_default = True

    def __call__(self, monomial):
        return monomial
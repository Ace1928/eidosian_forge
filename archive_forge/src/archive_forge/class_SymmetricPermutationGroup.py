from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
class SymmetricPermutationGroup(Basic):
    """
    The class defining the lazy form of SymmetricGroup.

    deg : int

    """

    def __new__(cls, deg):
        deg = _sympify(deg)
        obj = Basic.__new__(cls, deg)
        return obj

    def __init__(self, *args, **kwargs):
        self._deg = self.args[0]
        self._order = None

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> Permutation(1, 2, 3) in G
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError('A SymmetricPermutationGroup contains only Permutations as elements, not elements of type %s' % type(i))
        return i.size == self.degree

    def order(self):
        """
        Return the order of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.order()
        24
        """
        if self._order is not None:
            return self._order
        n = self._deg
        self._order = factorial(n)
        return self._order

    @property
    def degree(self):
        """
        Return the degree of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.degree
        4

        """
        return self._deg

    @property
    def identity(self):
        """
        Return the identity element of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.identity()
        (3)

        """
        return _af_new(list(range(self._deg)))
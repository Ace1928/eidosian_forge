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
def abelian_invariants(self):
    """
        Returns the abelian invariants for the given group.
        Let ``G`` be a nontrivial finite abelian group. Then G is isomorphic to
        the direct product of finitely many nontrivial cyclic groups of
        prime-power order.

        Explanation
        ===========

        The prime-powers that occur as the orders of the factors are uniquely
        determined by G. More precisely, the primes that occur in the orders of the
        factors in any such decomposition of ``G`` are exactly the primes that divide
        ``|G|`` and for any such prime ``p``, if the orders of the factors that are
        p-groups in one such decomposition of ``G`` are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``,
        then the orders of the factors that are p-groups in any such decomposition of ``G``
        are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``.

        The uniquely determined integers ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``, taken
        for all primes that divide ``|G|`` are called the invariants of the nontrivial
        group ``G`` as suggested in ([14], p. 542).

        Notes
        =====

        We adopt the convention that the invariants of a trivial group are [].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.abelian_invariants()
        [2]
        >>> from sympy.combinatorics import CyclicGroup
        >>> G = CyclicGroup(7)
        >>> G.abelian_invariants()
        [7]

        """
    if self.is_trivial:
        return []
    gns = self.generators
    inv = []
    G = self
    H = G.derived_subgroup()
    Hgens = H.generators
    for p in primefactors(G.order()):
        ranks = []
        while True:
            pows = []
            for g in gns:
                elm = g ** p
                if not H.contains(elm):
                    pows.append(elm)
            K = PermutationGroup(Hgens + pows) if pows else H
            r = G.order() // K.order()
            G = K
            gns = pows
            if r == 1:
                break
            ranks.append(multiplicity(p, r))
        if ranks:
            pows = [1] * ranks[0]
            for i in ranks:
                for j in range(i):
                    pows[j] = pows[j] * p
            inv.extend(pows)
    inv.sort()
    return inv
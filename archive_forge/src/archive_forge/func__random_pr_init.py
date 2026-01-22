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
def _random_pr_init(self, r, n, _random_prec_n=None):
    """Initialize random generators for the product replacement algorithm.

        Explanation
        ===========

        The implementation uses a modification of the original product
        replacement algorithm due to Leedham-Green, as described in [1],
        pp. 69-71; also, see [2], pp. 27-29 for a detailed theoretical
        analysis of the original product replacement algorithm, and [4].

        The product replacement algorithm is used for producing random,
        uniformly distributed elements of a group `G` with a set of generators
        `S`. For the initialization ``_random_pr_init``, a list ``R`` of
        `\\max\\{r, |S|\\}` group generators is created as the attribute
        ``G._random_gens``, repeating elements of `S` if necessary, and the
        identity element of `G` is appended to ``R`` - we shall refer to this
        last element as the accumulator. Then the function ``random_pr()``
        is called ``n`` times, randomizing the list ``R`` while preserving
        the generation of `G` by ``R``. The function ``random_pr()`` itself
        takes two random elements ``g, h`` among all elements of ``R`` but
        the accumulator and replaces ``g`` with a randomly chosen element
        from `\\{gh, g(~h), hg, (~h)g\\}`. Then the accumulator is multiplied
        by whatever ``g`` was replaced by. The new value of the accumulator is
        then returned by ``random_pr()``.

        The elements returned will eventually (for ``n`` large enough) become
        uniformly distributed across `G` ([5]). For practical purposes however,
        the values ``n = 50, r = 11`` are suggested in [1].

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: it changes the attribute
        self._random_gens

        See Also
        ========

        random_pr

        """
    deg = self.degree
    random_gens = [x._array_form for x in self.generators]
    k = len(random_gens)
    if k < r:
        for i in range(k, r):
            random_gens.append(random_gens[i - k])
    acc = list(range(deg))
    random_gens.append(acc)
    self._random_gens = random_gens
    if _random_prec_n is None:
        for i in range(n):
            self.random_pr()
    else:
        for i in range(n):
            self.random_pr(_random_prec=_random_prec_n[i])
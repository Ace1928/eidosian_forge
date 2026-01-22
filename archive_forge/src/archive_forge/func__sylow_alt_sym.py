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
def _sylow_alt_sym(self, p):
    """
        Return a p-Sylow subgroup of a symmetric or an
        alternating group.

        Explanation
        ===========

        The algorithm for this is hinted at in [1], Chapter 4,
        Exercise 4.

        For Sym(n) with n = p^i, the idea is as follows. Partition
        the interval [0..n-1] into p equal parts, each of length p^(i-1):
        [0..p^(i-1)-1], [p^(i-1)..2*p^(i-1)-1]...[(p-1)*p^(i-1)..p^i-1].
        Find a p-Sylow subgroup of Sym(p^(i-1)) (treated as a subgroup
        of ``self``) acting on each of the parts. Call the subgroups
        P_1, P_2...P_p. The generators for the subgroups P_2...P_p
        can be obtained from those of P_1 by applying a "shifting"
        permutation to them, that is, a permutation mapping [0..p^(i-1)-1]
        to the second part (the other parts are obtained by using the shift
        multiple times). The union of this permutation and the generators
        of P_1 is a p-Sylow subgroup of ``self``.

        For n not equal to a power of p, partition
        [0..n-1] in accordance with how n would be written in base p.
        E.g. for p=2 and n=11, 11 = 2^3 + 2^2 + 1 so the partition
        is [[0..7], [8..9], {10}]. To generate a p-Sylow subgroup,
        take the union of the generators for each of the parts.
        For the above example, {(0 1), (0 2)(1 3), (0 4), (1 5)(2 7)}
        from the first part, {(8 9)} from the second part and
        nothing from the third. This gives 4 generators in total, and
        the subgroup they generate is p-Sylow.

        Alternating groups are treated the same except when p=2. In this
        case, (0 1)(s s+1) should be added for an appropriate s (the start
        of a part) for each part in the partitions.

        See Also
        ========

        sylow_subgroup, is_alt_sym

        """
    n = self.degree
    gens = []
    identity = Permutation(n - 1)
    alt = p == 2 and all((g.is_even for g in self.generators))
    coeffs = []
    m = n
    while m > 0:
        coeffs.append(m % p)
        m = m // p
    power = len(coeffs) - 1
    for i in range(1, power + 1):
        if i == 1 and alt:
            continue
        gen = Permutation([(j + p ** (i - 1)) % p ** i for j in range(p ** i)])
        gens.append(identity * gen)
        if alt:
            gen = Permutation(0, 1) * gen * Permutation(0, 1) * gen
            gens.append(gen)
    start = 0
    while power > 0:
        a = coeffs[power]
        for _ in range(a):
            shift = Permutation()
            if start > 0:
                for i in range(p ** power):
                    shift = shift(i, start + i)
                if alt:
                    gen = Permutation(0, 1) * shift * Permutation(0, 1) * shift
                    gens.append(gen)
                    j = 2 * (power - 1)
                else:
                    j = power
                for i, gen in enumerate(gens[:j]):
                    if alt and i % 2 == 1:
                        continue
                    gen = shift * gen * shift
                    gens.append(gen)
            start += p ** power
        power = power - 1
    return gens
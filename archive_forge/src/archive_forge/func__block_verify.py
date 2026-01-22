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
def _block_verify(self, L, alpha):
    delta = sorted(self.orbit(alpha))
    p = [-1] * len(delta)
    blocks = [-1] * len(delta)
    B = [[]]
    u = [0] * len(delta)
    t = L.orbit_transversal(alpha, pairs=True)
    for a, beta in t:
        B[0].append(a)
        i_a = delta.index(a)
        p[i_a] = 0
        blocks[i_a] = alpha
        u[i_a] = beta
    rho = 0
    m = 0
    while rho <= m:
        beta = B[rho][0]
        for g in self.generators:
            d = beta ^ g
            i_d = delta.index(d)
            sigma = p[i_d]
            if sigma < 0:
                m += 1
                sigma = m
                u[i_d] = u[delta.index(beta)] * g
                p[i_d] = sigma
                rep = d
                blocks[i_d] = rep
                newb = [rep]
                for gamma in B[rho][1:]:
                    i_gamma = delta.index(gamma)
                    d = gamma ^ g
                    i_d = delta.index(d)
                    if p[i_d] < 0:
                        u[i_d] = u[i_gamma] * g
                        p[i_d] = sigma
                        blocks[i_d] = rep
                        newb.append(d)
                    else:
                        s = u[i_gamma] * g * u[i_d] ** (-1)
                        return (False, s)
                B.append(newb)
            else:
                for h in B[rho][1:]:
                    if h ^ g not in B[sigma]:
                        s = u[delta.index(beta)] * g * u[i_d] ** (-1)
                        return (False, s)
        rho += 1
    return (True, blocks)
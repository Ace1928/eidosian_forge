from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _initialize_ith_poly(N, factor_base, i, g, B):
    """Initialization stage of ith poly. After we finish sieving 1`st polynomial
    here we quickly change to the next polynomial from which we will again
    start sieving. Suppose we generated ith sieve polynomial and now we
    want to generate (i + 1)th polynomial, where ``1 <= i <= 2**(j - 1) - 1``
    where `j` is the number of prime factors of the coefficient `a`
    then this function can be used to go to the next polynomial. If
    ``i = 2**(j - 1) - 1`` then go to _initialize_first_polynomial stage.

    Parameters
    ==========

    N : number to be factored
    factor_base : factor_base primes
    i : integer denoting ith polynomial
    g : (i - 1)th polynomial
    B : array that stores a//q_l*gamma
    """
    from sympy.functions.elementary.integers import ceiling
    v = 1
    j = i
    while j % 2 == 0:
        v += 1
        j //= 2
    if ceiling(i / 2 ** v) % 2 == 1:
        neg_pow = -1
    else:
        neg_pow = 1
    b = g.b + 2 * neg_pow * B[v - 1]
    a = g.a
    g = SievePolynomial([a * a, 2 * a * b, b * b - N], a, b)
    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.soln1 = (fb.soln1 - neg_pow * fb.b_ainv[v - 1]) % fb.prime
        fb.soln2 = (fb.soln2 - neg_pow * fb.b_ainv[v - 1]) % fb.prime
    return g
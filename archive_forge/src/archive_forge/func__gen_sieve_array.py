from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _gen_sieve_array(M, factor_base):
    """Sieve Stage of the Quadratic Sieve. For every prime in the factor_base
    that does not divide the coefficient `a` we add log_p over the sieve_array
    such that ``-M <= soln1 + i*p <=  M`` and ``-M <= soln2 + i*p <=  M`` where `i`
    is an integer. When p = 2 then log_p is only added using
    ``-M <= soln1 + i*p <=  M``.

    Parameters
    ==========

    M : sieve interval
    factor_base : factor_base primes
    """
    sieve_array = [0] * (2 * M + 1)
    for factor in factor_base:
        if factor.soln1 is None:
            continue
        for idx in range((M + factor.soln1) % factor.prime, 2 * M, factor.prime):
            sieve_array[idx] += factor.log_p
        if factor.prime == 2:
            continue
        for idx in range((M + factor.soln2) % factor.prime, 2 * M, factor.prime):
            sieve_array[idx] += factor.log_p
    return sieve_array
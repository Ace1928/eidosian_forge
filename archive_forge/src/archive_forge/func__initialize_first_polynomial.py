from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _initialize_first_polynomial(N, M, factor_base, idx_1000, idx_5000, seed=None):
    """This step is the initialization of the 1st sieve polynomial.
    Here `a` is selected as a product of several primes of the factor_base
    such that `a` is about to ``sqrt(2*N) / M``. Other initial values of
    factor_base elem are also initialized which includes a_inv, b_ainv, soln1,
    soln2 which are used when the sieve polynomial is changed. The b_ainv
    is required for fast polynomial change as we do not have to calculate
    `2*b*mod_inverse(a, prime)` every time.
    We also ensure that the `factor_base` primes which make `a` are between
    1000 and 5000.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    seed : Generate pseudoprime numbers
    """
    if seed is not None:
        rgen.seed(seed)
    approx_val = sqrt(2 * N) / M
    best_a, best_q, best_ratio = (None, None, None)
    start = 0 if idx_1000 is None else idx_1000
    end = len(factor_base) - 1 if idx_5000 is None else idx_5000
    for _ in range(50):
        a = 1
        q = []
        while a < approx_val:
            rand_p = 0
            while rand_p == 0 or rand_p in q:
                rand_p = rgen.randint(start, end)
            p = factor_base[rand_p].prime
            a *= p
            q.append(rand_p)
        ratio = a / approx_val
        if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
            best_q = q
            best_a = a
            best_ratio = ratio
    a = best_a
    q = best_q
    B = []
    for idx, val in enumerate(q):
        q_l = factor_base[val].prime
        gamma = factor_base[val].tmem_p * mod_inverse(a // q_l, q_l) % q_l
        if gamma > q_l / 2:
            gamma = q_l - gamma
        B.append(a // q_l * gamma)
    b = sum(B)
    g = SievePolynomial([a * a, 2 * a * b, b * b - N], a, b)
    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.a_inv = mod_inverse(a, fb.prime)
        fb.b_ainv = [2 * b_elem * fb.a_inv % fb.prime for b_elem in B]
        fb.soln1 = fb.a_inv * (fb.tmem_p - b) % fb.prime
        fb.soln2 = fb.a_inv * (-fb.tmem_p - b) % fb.prime
    return (g, B)
from mpmath.libmp import (fzero, from_int, from_rational,
from sympy.core.numbers import igcd
from .residue_ntheory import (_sqrt_mod_prime_power,
import math
def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    References
    ==========

    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
    if k == 1:
        return fone
    k1 = k
    e = 0
    p = _factor[k]
    while k1 % p == 0:
        k1 //= p
        e += 1
    k2 = k // k1
    v = 1 - 24 * n
    pi = mpf_pi(prec)
    if k1 == 1:
        if p == 2:
            mod = 8 * k
            v = mod + v % mod
            v = v * pow(9, k - 1, mod) % mod
            m = _sqrt_mod_prime_power(v, 2, e + 3)[0]
            arg = mpf_div(mpf_mul(from_int(4 * m), pi, prec), from_int(mod), prec)
            return mpf_mul(mpf_mul(from_int((-1) ** e * jacobi_symbol(m - 1, m)), mpf_sqrt(from_int(k), prec), prec), mpf_sin(arg, prec), prec)
        if p == 3:
            mod = 3 * k
            v = mod + v % mod
            if e > 1:
                v = v * pow(64, k // 3 - 1, mod) % mod
            m = _sqrt_mod_prime_power(v, 3, e + 1)[0]
            arg = mpf_div(mpf_mul(from_int(4 * m), pi, prec), from_int(mod), prec)
            return mpf_mul(mpf_mul(from_int(2 * (-1) ** (e + 1) * legendre_symbol(m, 3)), mpf_sqrt(from_int(k // 3), prec), prec), mpf_sin(arg, prec), prec)
        v = k + v % k
        if v % p == 0:
            if e == 1:
                return mpf_mul(from_int(jacobi_symbol(3, k)), mpf_sqrt(from_int(k), prec), prec)
            return fzero
        if not is_quad_residue(v, p):
            return fzero
        _phi = p ** (e - 1) * (p - 1)
        v = v * pow(576, _phi - 1, k)
        m = _sqrt_mod_prime_power(v, p, e)[0]
        arg = mpf_div(mpf_mul(from_int(4 * m), pi, prec), from_int(k), prec)
        return mpf_mul(mpf_mul(from_int(2 * jacobi_symbol(3, k)), mpf_sqrt(from_int(k), prec), prec), mpf_cos(arg, prec), prec)
    if p != 2 or e >= 3:
        d1, d2 = (igcd(k1, 24), igcd(k2, 24))
        e = 24 // (d1 * d2)
        n1 = (d2 * e * n + (k2 ** 2 - 1) // d1) * pow(e * k2 * k2 * d2, _totient[k1] - 1, k1) % k1
        n2 = (d1 * e * n + (k1 ** 2 - 1) // d2) * pow(e * k1 * k1 * d1, _totient[k2] - 1, k2) % k2
        return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
    if e == 2:
        n1 = (8 * n + 5) * pow(128, _totient[k1] - 1, k1) % k1
        n2 = (4 + (n - 2 - (k1 ** 2 - 1) // 8) * k1 ** 2 % 4) % 4
        return mpf_mul(mpf_mul(from_int(-1), _a(n1, k1, prec), prec), _a(n2, k2, prec))
    n1 = (8 * n + 1) * pow(32, _totient[k1] - 1, k1) % k1
    n2 = (2 + (n - (k1 ** 2 - 1) // 8) % 2) % 2
    return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
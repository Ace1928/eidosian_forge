from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def _sqrt_mod_tonelli_shanks(a, p):
    """
    Returns the square root in the case of ``p`` prime with ``p == 1 (mod 8)``

    References
    ==========

    .. [1] R. Crandall and C. Pomerance "Prime Numbers", 2nt Ed., page 101

    """
    s = trailing(p - 1)
    t = p >> s
    while 1:
        d = randint(2, p - 1)
        r = legendre_symbol(d, p)
        if r == -1:
            break
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        adm = A * pow(D, m, p) % p
        adm = pow(adm, 2 ** (s - 1 - i), p)
        if adm % p == p - 1:
            m += 2 ** i
    x = pow(a, (t + 1) // 2, p) * pow(D, m // 2, p) % p
    return x
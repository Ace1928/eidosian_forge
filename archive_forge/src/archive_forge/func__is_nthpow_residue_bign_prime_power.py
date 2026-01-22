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
def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    """Returns True/False if a solution for `x^n = a \\pmod{p^k}`
    does/does not exist."""
    if a % p:
        if p != 2:
            return _is_nthpow_residue_bign(a, n, pow(p, k))
        if n & 1:
            return True
        c = trailing(n)
        return a % pow(2, min(c + 2, k)) == 1
    else:
        a %= pow(p, k)
        if not a:
            return True
        mu = multiplicity(p, a)
        if mu % n:
            return False
        pm = pow(p, mu)
        return _is_nthpow_residue_bign_prime_power(a // pm, n, p, k - mu)
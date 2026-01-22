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
def _nthroot_mod_composite(a, n, m):
    """
    Find the solutions to ``x**n = a mod m`` when m is not prime.
    """
    return _help(m, lambda p: nthroot_mod(a, n, p, True), lambda root, p: pow(root, n - 1, p) * (n % p) % p, lambda root, p: (pow(root, n, p) - a) % p)
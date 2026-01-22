from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _gf_gcdex(f, g, p):
    """
    Extended Euclidean Algorithm for two univariate polynomials over
    `\\mathbb Z_p`.

    Returns polynomials `s, t` and `h`, such that `h` is the GCD of `f` and
    `g` and `sf + tg = h \\; \\mathrm{mod} \\, p`.

    """
    ring = f.ring
    s, t, h = gf_gcdex(f.to_dense(), g.to_dense(), p, ring.domain)
    return (ring.from_dense(s), ring.from_dense(t), ring.from_dense(h))
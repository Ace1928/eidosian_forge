from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import ModularGCDFailed
from mpmath import sqrt
import random
def _trivial_gcd(f, g):
    """
    Compute the GCD of two polynomials in trivial cases, i.e. when one
    or both polynomials are zero.
    """
    ring = f.ring
    if not (f or g):
        return (ring.zero, ring.zero, ring.zero)
    elif not f:
        if g.LC < ring.domain.zero:
            return (-g, ring.zero, -ring.one)
        else:
            return (g, ring.zero, ring.one)
    elif not g:
        if f.LC < ring.domain.zero:
            return (-f, -ring.one, ring.zero)
        else:
            return (f, ring.one, ring.zero)
    return None
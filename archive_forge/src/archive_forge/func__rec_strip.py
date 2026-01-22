from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_strip(g, v):
    """Recursive helper for :func:`_rec_strip`."""
    if not v:
        return dup_strip(g)
    w = v - 1
    return dmp_strip([_rec_strip(c, w) for c in g], v)
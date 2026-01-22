from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def _rec_list_terms(g, v, monom):
    """Recursive helper for :func:`dmp_list_terms`."""
    d, terms = (dmp_degree(g, v), [])
    if not v:
        for i, c in enumerate(g):
            if not c:
                continue
            terms.append((monom + (d - i,), c))
    else:
        w = v - 1
        for i, c in enumerate(g):
            terms.extend(_rec_list_terms(c, w, monom + (d - i,)))
    return terms
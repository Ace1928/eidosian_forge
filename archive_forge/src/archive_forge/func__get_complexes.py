from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
@classmethod
def _get_complexes(cls, factors, use_cache=True):
    """Compute complex root isolating intervals for a list of factors. """
    complexes = []
    for currentfactor, k in ordered(factors):
        try:
            if not use_cache:
                raise KeyError
            c = _complexes_cache[currentfactor]
            complexes.extend([(i, currentfactor, k) for i in c])
        except KeyError:
            complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
            new = [(root, currentfactor, k) for root in complex_part]
            complexes.extend(new)
    complexes = cls._complexes_sorted(complexes)
    return complexes
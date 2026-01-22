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
def _indexed_root(cls, poly, index, lazy=False):
    """Get a root of a composite polynomial by index. """
    factors = _pure_factors(poly)
    if lazy and len(factors) == 1 and (factors[0][1] == 1):
        return (poly, index)
    reals = cls._get_reals(factors)
    reals_count = cls._count_roots(reals)
    if index < reals_count:
        return cls._reals_index(reals, index)
    else:
        complexes = cls._get_complexes(factors)
        return cls._complexes_index(complexes, index - reals_count)
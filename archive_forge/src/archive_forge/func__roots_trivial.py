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
@cacheit
def _roots_trivial(cls, poly, radicals):
    """Compute roots in linear, quadratic and binomial cases. """
    if poly.degree() == 1:
        return roots_linear(poly)
    if not radicals:
        return None
    if poly.degree() == 2:
        return roots_quadratic(poly)
    elif poly.length() == 2 and poly.TC():
        return roots_binomial(poly)
    else:
        return None
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
def _refine_complexes(cls, complexes):
    """return complexes such that no bounding rectangles of non-conjugate
        roots would intersect. In addition, assure that neither ay nor by is
        0 to guarantee that non-real roots are distinct from real roots in
        terms of the y-bounds.
        """
    for i, (u, f, k) in enumerate(complexes):
        for j, (v, g, m) in enumerate(complexes[i + 1:]):
            u, v = u.refine_disjoint(v)
            complexes[i + j + 1] = (v, g, m)
        complexes[i] = (u, f, k)
    complexes = cls._refine_imaginary(complexes)
    for i, (u, f, k) in enumerate(complexes):
        while u.ay * u.by <= 0:
            u = u.refine()
        complexes[i] = (u, f, k)
    return complexes
from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _compute_fourth_corner(T):
    v = 4 * [None]
    missing_corners = [V for V in ZeroSubsimplices if T.Class[V].IdealPoint is None]
    if not missing_corners:
        return
    missing_corner = missing_corners[0]
    v[3] = missing_corner
    v[0] = ([V for V in ZeroSubsimplices if T.Class[V].IdealPoint == Infinity] + [V for V in ZeroSubsimplices if V != missing_corner])[0]
    v[1], v[2] = (_RemainingFace[v[3], v[0]], _RemainingFace[v[0], v[3]])
    z = [T.Class[V].IdealPoint for V in v]
    cross_ratio = T.ShapeParameters[v[0] | v[1]]
    if z[0] == Infinity:
        z[3] = z[1] + cross_ratio * (z[2] - z[1])
    else:
        diff20 = z[2] - z[0]
        diff21 = z[2] - z[1]
        numerator = z[1] * diff20 - cross_ratio * (z[0] * diff21)
        denominator = diff20 - cross_ratio * diff21
        if abs(denominator) == 0 and abs(numerator) > 0:
            z[3] = Infinity
        else:
            z[3] = numerator / denominator
    T.Class[missing_corner].IdealPoint = z[3]
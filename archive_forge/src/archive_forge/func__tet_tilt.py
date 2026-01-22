from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _tet_tilt(tet, face):
    """The tilt of the face of the tetrahedron."""
    v = t3m.simplex.comp(face)
    ans = 0
    for w in t3m.simplex.ZeroSubsimplices:
        if v == w:
            c_w = 1
        else:
            z = tet.ShapeParameters[v | w]
            c_w = -z.real() / abs(z)
        R_w = tet.horotriangles[w].circumradius
        ans += c_w * R_w
    return ans
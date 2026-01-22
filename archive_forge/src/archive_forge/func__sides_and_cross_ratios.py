from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def _sides_and_cross_ratios(tet, vertex, side):
    sides = t3m.simplex.FacesAroundVertexCounterclockwise[vertex]
    left_side, center_side, right_side = HoroTriangleBase._make_second(sides, side)
    z_left = tet.ShapeParameters[left_side & center_side]
    z_right = tet.ShapeParameters[center_side & right_side]
    return (left_side, center_side, right_side, z_left, z_right)
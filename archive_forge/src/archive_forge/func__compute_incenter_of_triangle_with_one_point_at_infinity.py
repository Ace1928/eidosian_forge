from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
def _compute_incenter_of_triangle_with_one_point_at_infinity(nonInfPoints):
    a, b = nonInfPoints
    RIF = a.real().parent()
    return FinitePoint((a + b) / 2, abs(a - b) * RIF(3).sqrt() / 2)
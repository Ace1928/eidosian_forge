from fontTools.misc.transform import Identity, Scale
from math import atan2, ceil, cos, fabs, isfinite, pi, radians, sin, sqrt, tan
def _map_point(matrix, pt):
    r = matrix.transformPoint((pt.real, pt.imag))
    return r[0] + r[1] * 1j
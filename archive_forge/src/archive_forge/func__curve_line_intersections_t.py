from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _curve_line_intersections_t(curve, line):
    aligned_curve = _alignment_transformation(line).transformPoints(curve)
    if len(curve) == 3:
        a, b, c = calcQuadraticParameters(*aligned_curve)
        intersections = solveQuadratic(a[1], b[1], c[1])
    elif len(curve) == 4:
        a, b, c, d = calcCubicParameters(*aligned_curve)
        intersections = solveCubic(a[1], b[1], c[1], d[1])
    else:
        raise ValueError('Unknown curve degree')
    return sorted((i for i in intersections if 0.0 <= i <= 1))
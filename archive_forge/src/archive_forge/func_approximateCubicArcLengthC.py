from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.double)
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, pt4=cython.complex)
@cython.locals(v0=cython.double, v1=cython.double, v2=cython.double, v3=cython.double, v4=cython.double)
def approximateCubicArcLengthC(pt1, pt2, pt3, pt4):
    """Approximates the arc length for a cubic Bezier segment.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers.

    Returns:
        Arc length value.
    """
    v0 = abs(pt2 - pt1) * 0.15
    v1 = abs(-0.558983582205757 * pt1 + 0.325650248872424 * pt2 + 0.208983582205757 * pt3 + 0.024349751127576 * pt4)
    v2 = abs(pt4 - pt1 + pt3 - pt2) * 0.26666666666666666
    v3 = abs(-0.024349751127576 * pt1 - 0.208983582205757 * pt2 - 0.325650248872424 * pt3 + 0.558983582205757 * pt4)
    v4 = abs(pt4 - pt3) * 0.15
    return v0 + v1 + v2 + v3 + v4
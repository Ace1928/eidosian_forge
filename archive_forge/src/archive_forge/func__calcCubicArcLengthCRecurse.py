from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.returns(cython.double)
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
@cython.locals(mult=cython.double, arch=cython.double, box=cython.double)
def _calcCubicArcLengthCRecurse(mult, p0, p1, p2, p3):
    arch = abs(p0 - p3)
    box = abs(p0 - p1) + abs(p1 - p2) + abs(p2 - p3)
    if arch * mult >= box:
        return (arch + box) * 0.5
    else:
        one, two = _split_cubic_into_two(p0, p1, p2, p3)
        return _calcCubicArcLengthCRecurse(mult, *one) + _calcCubicArcLengthCRecurse(mult, *two)
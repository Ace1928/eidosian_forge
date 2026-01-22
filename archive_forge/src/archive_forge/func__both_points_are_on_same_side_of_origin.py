from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _both_points_are_on_same_side_of_origin(a, b, origin):
    xDiff = (a[0] - origin[0]) * (b[0] - origin[0])
    yDiff = (a[1] - origin[1]) * (b[1] - origin[1])
    return not (xDiff <= 0.0 and yDiff <= 0.0)
from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _line_t_of_pt(s, e, pt):
    sx, sy = s
    ex, ey = e
    px, py = pt
    if abs(sx - ex) < epsilon and abs(sy - ey) < epsilon:
        return -1
    if abs(sx - ex) > abs(sy - ey):
        return (px - sx) / (ex - sx)
    else:
        return (py - sy) / (ey - sy)
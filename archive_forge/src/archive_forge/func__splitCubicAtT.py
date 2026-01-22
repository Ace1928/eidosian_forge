from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def _splitCubicAtT(a, b, c, d, *ts):
    ts = list(ts)
    ts.insert(0, 0.0)
    ts.append(1.0)
    segments = []
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    for i in range(len(ts) - 1):
        t1 = ts[i]
        t2 = ts[i + 1]
        delta = t2 - t1
        delta_2 = delta * delta
        delta_3 = delta * delta_2
        t1_2 = t1 * t1
        t1_3 = t1 * t1_2
        a1x = ax * delta_3
        a1y = ay * delta_3
        b1x = (3 * ax * t1 + bx) * delta_2
        b1y = (3 * ay * t1 + by) * delta_2
        c1x = (2 * bx * t1 + cx + 3 * ax * t1_2) * delta
        c1y = (2 * by * t1 + cy + 3 * ay * t1_2) * delta
        d1x = ax * t1_3 + bx * t1_2 + cx * t1 + dx
        d1y = ay * t1_3 + by * t1_2 + cy * t1 + dy
        pt1, pt2, pt3, pt4 = calcCubicPoints((a1x, a1y), (b1x, b1y), (c1x, c1y), (d1x, d1y))
        segments.append((pt1, pt2, pt3, pt4))
    return segments
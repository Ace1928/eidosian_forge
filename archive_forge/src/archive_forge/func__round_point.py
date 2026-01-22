from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound
def _round_point(pt):
    return (otRound(pt[0]), otRound(pt[1]))
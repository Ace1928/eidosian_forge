from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (

    q: quadratic spline with alternating on-curve / off-curve points.

    costs: cumulative list of encoding cost of q in terms of number of
      points that need to be encoded.  Implied on-curve points do not
      contribute to the cost. If all points need to be encoded, then
      costs will be range(1, len(q)+1).
    
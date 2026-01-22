from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import (
import math
def _addCubicQuadrature(self, c0, c1, c2, c3):
    self.value += approximateCubicArcLengthC(c0, c1, c2, c3)
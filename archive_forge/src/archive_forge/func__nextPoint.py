from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def _nextPoint(self, point):
    x, y = self.currentPoint
    point = (x + point[0], y + point[1])
    self.currentPoint = point
    return point
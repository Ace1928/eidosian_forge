from fontTools.misc.arrayTools import updateBounds, pointInRect, unionRect
from fontTools.misc.bezierTools import calcCubicBounds, calcQuadraticBounds
from fontTools.pens.basePen import BasePen
def _addMoveTo(self):
    if self._start is None:
        return
    bounds = self.bounds
    if bounds:
        self.bounds = updateBounds(bounds, self._start)
    else:
        x, y = self._start
        self.bounds = (x, y, x, y)
    self._start = None
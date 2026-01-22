from fontTools.pens.basePen import BasePen, OpenContourError
def _closePath(self):
    p0 = self._getCurrentPoint()
    if p0 != self.__startPoint:
        self._lineTo(self.__startPoint)
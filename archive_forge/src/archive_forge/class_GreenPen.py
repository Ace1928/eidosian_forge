from fontTools.pens.basePen import BasePen
from functools import partial
from itertools import count
import sympy as sp
import sys
class GreenPen(BasePen):
    _BezierFuncs = {}

    @classmethod
    def _getGreenBezierFuncs(celf, func):
        funcstr = str(func)
        if not funcstr in celf._BezierFuncs:
            celf._BezierFuncs[funcstr] = _BezierFuncsLazy(func)
        return celf._BezierFuncs[funcstr]

    def __init__(self, func, glyphset=None):
        BasePen.__init__(self, glyphset)
        self._funcs = self._getGreenBezierFuncs(func)
        self.value = 0

    def _moveTo(self, p0):
        self.__startPoint = p0

    def _closePath(self):
        p0 = self._getCurrentPoint()
        if p0 != self.__startPoint:
            self._lineTo(self.__startPoint)

    def _endPath(self):
        p0 = self._getCurrentPoint()
        if p0 != self.__startPoint:
            raise NotImplementedError

    def _lineTo(self, p1):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[1](p0, p1)

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[2](p0, p1, p2)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[3](p0, p1, p2, p3)
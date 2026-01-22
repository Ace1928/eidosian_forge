from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
class _XTicks:
    _tickTweaks = 0

    @property
    def actualTickStrokeWidth(self):
        return getattr(self, 'tickStrokeWidth', self.strokeWidth)

    @property
    def actualTickStrokeColor(self):
        return getattr(self, 'tickStrokeColor', self.strokeColor)

    def _drawTicksInner(self, tU, tD, g):
        itd = getattr(self, 'innerTickDraw', None)
        if itd:
            itd(self, tU, tD, g)
        elif tU or tD:
            sW = self.actualTickStrokeWidth
            tW = self._tickTweaks
            if tW:
                if tU and (not tD):
                    tD = tW * sW
                elif tD and (not tU):
                    tU = tW * sW
            self._makeLines(g, tU, -tD, self.actualTickStrokeColor, sW, self.strokeDashArray, self.strokeLineJoin, self.strokeLineCap, self.strokeMiterLimit)

    def _drawTicks(self, tU, tD, g=None):
        g = g or Group()
        if self.visibleTicks:
            self._drawTicksInner(tU, tD, g)
        return g

    def _drawSubTicks(self, tU, tD, g):
        if getattr(self, 'visibleSubTicks', 0) and self.subTickNum > 0:
            otv = self._calcSubTicks()
            try:
                self._subTicking = 1
                self._drawTicksInner(tU, tD, g)
            finally:
                del self._subTicking
                self._tickValues = otv

    def makeTicks(self):
        yold = self._y
        try:
            self._y = self._labelAxisPos(getattr(self, 'tickAxisMode', 'axis'))
            g = self._drawTicks(self.tickUp, self.tickDown)
            self._drawSubTicks(getattr(self, 'subTickHi', 0), getattr(self, 'subTickLo', 0), g)
            return g
        finally:
            self._y = yold

    def _labelAxisPos(self, mode=None):
        axis = self.joinAxis
        if axis:
            mode = mode or self.labelAxisMode
            if mode == 'low':
                return axis._y
            elif mode == 'high':
                return axis._y + axis._length
        return self._y
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
def _setRange(self, dataSeries):
    valueMin = self.valueMin
    valueMax = self.valueMax
    aMin = _findMin(dataSeries, self._dataIndex, 0, extraMinMaxValues=self.extraMinMaxValues)
    aMax = _findMax(dataSeries, self._dataIndex, 0, extraMinMaxValues=self.extraMinMaxValues)
    if valueMin is None:
        valueMin = aMin
    if valueMax is None:
        valueMax = aMax
    if valueMin > valueMax:
        raise ValueError('%s: valueMin=%r should not be greater than valueMax=%r!' % (self.__class__.__name__valueMin, valueMax))
    if valueMin <= 0:
        raise ValueError('%s: valueMin=%r negative values are not allowed!' % (self.__class__.__name__, valueMin))
    abS = self.avoidBoundSpace
    if abS:
        lMin = math_log10(aMin)
        lMax = math_log10(aMax)
        if not isSeq(abS):
            abS = (abS, abS)
        a0 = abS[0] or 0
        a1 = abS[1] or 0
        L = self._length - (a0 + a1)
        sf = (lMax - lMin) / float(L)
        lMin -= a0 * sf
        lMax += a1 * sf
        valueMin = min(valueMin, 10 ** lMin)
        valueMax = max(valueMax, 10 ** lMax)
    self._valueMin = valueMin
    self._valueMax = valueMax
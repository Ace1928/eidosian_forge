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
class LogValueAxis(ValueAxis):

    def _calcScaleFactor(self):
        """Calculate the axis' scale factor.
        This should be called only *after* the axis' range is set.
        Returns a number.
        """
        self._scaleFactor = self._length / float(math_log10(self._valueMax) - math_log10(self._valueMin))
        return self._scaleFactor

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

    def _calcTickPositions(self):
        valueMin = cMin = math_log10(self._valueMin)
        valueMax = cMax = math_log10(self._valueMax)
        rr = self.rangeRound
        if rr:
            if rr in ('both', 'ceiling'):
                i = int(valueMax)
                valueMax = i + 1 if i < valueMax else i
            if rr in ('both', 'floor'):
                i = int(valueMin)
                valueMin = i - 1 if i > valueMin else i
        T = [].append
        tv = int(valueMin)
        if tv < valueMin:
            tv += 1
        n = int(valueMax) - tv + 1
        i = max(int(n / self.maximumTicks), 1)
        if i * n > self.maximumTicks:
            i += 1
        self._powerInc = i
        while True:
            if tv > valueMax:
                break
            if tv >= valueMin:
                T(10 ** tv)
            tv += i
        if valueMin != cMin:
            self._valueMin = 10 ** valueMin
        if valueMax != cMax:
            self._valueMax = 10 ** valueMax
        return T.__self__

    def _calcSubTicks(self):
        if not hasattr(self, '_tickValues'):
            self._pseudo_configure()
        otv = self._tickValues
        if not hasattr(self, '_subTickValues'):
            T = [].append
            valueMin = math_log10(self._valueMin)
            valueMax = math_log10(self._valueMax) + 1
            tv = round(valueMin)
            i = self._powerInc
            if i == 1:
                fac = 10 / float(self.subTickNum)
                start = 1
                if self.subTickNum == 10:
                    start = 2
                while tv < valueMax:
                    for j in range(start, self.subTickNum):
                        v = fac * j * 10 ** tv
                        if v > self._valueMin and v < self._valueMax:
                            T(v)
                    tv += i
            else:
                ng = min(self.subTickNum + 1, i - 1)
                while ng:
                    if i % ng == 0:
                        i /= ng
                        break
                    ng -= 1
                else:
                    i = 1
                tv = round(valueMin)
                while True:
                    v = 10 ** tv
                    if v >= self._valueMax:
                        break
                    if v not in otv:
                        T(v)
                    tv += i
            self._subTickValues = T.__self__
        self._tickValues = self._subTickValues
        return otv
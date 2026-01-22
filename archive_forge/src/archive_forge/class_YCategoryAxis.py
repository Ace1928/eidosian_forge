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
class YCategoryAxis(_YTicks, CategoryAxis):
    """Y/category axis"""
    _attrMap = AttrMap(BASE=CategoryAxis, tickLeft=AttrMapValue(isNumber, desc='Tick length left of the axis.'), tickRight=AttrMapValue(isNumber, desc='Tick length right of the axis.'), joinAxisMode=AttrMapValue(OneOf(('left', 'right', 'value', 'points', None)), desc="Mode used for connecting axis ('left', 'right', 'value', 'points', None)."))
    _dataIndex = 1

    def __init__(self):
        CategoryAxis.__init__(self)
        self.labels.boxAnchor = 'e'
        self.labels.dx = -5
        self.tickLeft = 5
        self.tickRight = 0

    def demo(self):
        self.setPosition(50, 10, 80)
        self.configure([(10, 20, 30)])
        self.categoryNames = ['One', 'Two', 'Three']
        self.labels.boxAnchor = 'e'
        self.labels[2].boxAnchor = 's'
        self.labels[2].angle = 90
        d = Drawing(200, 100)
        d.add(self)
        return d

    def joinToAxis(self, xAxis, mode='left', pos=None):
        """Join with x-axis using some mode."""
        _assertXAxis(xAxis)
        if mode == 'left':
            self._x = xAxis._x * 1.0
        elif mode == 'right':
            self._x = (xAxis._x + xAxis._length) * 1.0
        elif mode == 'value':
            self._x = xAxis.scale(pos) * 1.0
        elif mode == 'points':
            self._x = pos * 1.0

    def _joinToAxis(self):
        ja = self.joinAxis
        if ja:
            jam = self.joinAxisMode
            if jam in ('left', 'right'):
                self.joinToAxis(ja, mode=jam)
            elif jam in ('value', 'points'):
                self.joinToAxis(ja, mode=jam, pos=self.joinAxisPos)

    def loScale(self, idx):
        """Returns the y position in drawing units"""
        return self._y + self._scale(idx) * self._barWidth

    def makeAxis(self):
        g = Group()
        self._joinToAxis()
        if not self.visibleAxis:
            return g
        axis = Line(self._x, self._y - self.loLLen, self._x, self._y + self._length + self.hiLLen)
        axis.strokeColor = self.strokeColor
        axis.strokeWidth = self.strokeWidth
        axis.strokeDashArray = self.strokeDashArray
        g.add(axis)
        return g

    def makeTickLabels(self):
        g = Group()
        if not self.visibleLabels:
            return g
        categoryNames = self.categoryNames
        if categoryNames is not None:
            catCount = self._catCount
            n = len(categoryNames)
            reverseDirection = self.reverseDirection
            barWidth = self._barWidth
            labels = self.labels
            _x = self._labelAxisPos()
            _y = self._y
            pmv = self._pmv if self.labelAxisMode == 'axispmv' else None
            for i in range(catCount):
                if reverseDirection:
                    ic = catCount - i - 1
                else:
                    ic = i
                if ic >= n:
                    continue
                label = i - catCount
                if label in self.labels:
                    label = self.labels[label]
                else:
                    label = self.labels[i]
                lpf = label.labelPosFrac
                y = _y + (i + lpf) * barWidth
                if pmv:
                    _dx = label.dx
                    v = label._pmv = pmv[ic]
                    if v < 0:
                        _dx *= -2
                else:
                    _dx = 0
                label.setOrigin(_x + _dx, y)
                label.setText(categoryNames[ic] or '')
                g.add(label)
        return g
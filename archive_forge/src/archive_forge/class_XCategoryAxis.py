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
class XCategoryAxis(_XTicks, CategoryAxis):
    """X/category axis"""
    _attrMap = AttrMap(BASE=CategoryAxis, tickUp=AttrMapValue(isNumber, desc='Tick length up the axis.'), tickDown=AttrMapValue(isNumber, desc='Tick length down the axis.'), joinAxisMode=AttrMapValue(OneOf('bottom', 'top', 'value', 'points', None), desc="Mode used for connecting axis ('bottom', 'top', 'value', 'points', None)."))
    _dataIndex = 0

    def __init__(self):
        CategoryAxis.__init__(self)
        self.labels.boxAnchor = 'n'
        self.labels.dy = -5
        self.tickUp = 0
        self.tickDown = 5

    def demo(self):
        self.setPosition(30, 70, 140)
        self.configure([(10, 20, 30, 40, 50)])
        self.categoryNames = ['One', 'Two', 'Three', 'Four', 'Five']
        self.labels.boxAnchor = 'n'
        self.labels[4].boxAnchor = 'e'
        self.labels[4].angle = 90
        d = Drawing(200, 100)
        d.add(self)
        return d

    def joinToAxis(self, yAxis, mode='bottom', pos=None):
        """Join with y-axis using some mode."""
        _assertYAxis(yAxis)
        if mode == 'bottom':
            self._y = yAxis._y
        elif mode == 'top':
            self._y = yAxis._y + yAxis._length
        elif mode == 'value':
            self._y = yAxis.scale(pos)
        elif mode == 'points':
            self._y = pos

    def _joinToAxis(self):
        ja = self.joinAxis
        if ja:
            jam = self.joinAxisMode
            if jam in ('bottom', 'top'):
                self.joinToAxis(ja, mode=jam)
            elif jam in ('value', 'points'):
                self.joinToAxis(ja, mode=jam, pos=self.joinAxisPos)

    def loScale(self, idx):
        """returns the x position in drawing units"""
        return self._x + self.loPad + self._scale(idx) * self._barWidth

    def makeAxis(self):
        g = Group()
        self._joinToAxis()
        if not self.visibleAxis:
            return g
        axis = Line(self._x - self.loLLen, self._y, self._x + self._length + self.hiLLen, self._y)
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
            _y = self._labelAxisPos()
            _x = self._x
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
                if pmv:
                    _dy = label.dy
                    v = label._pmv = pmv[ic]
                    if v < 0:
                        _dy *= -2
                else:
                    _dy = 0
                lpf = label.labelPosFrac
                x = _x + (i + lpf) * barWidth
                label.setOrigin(x, _y + _dy)
                label.setText(categoryNames[ic] or '')
                g.add(label)
        return g
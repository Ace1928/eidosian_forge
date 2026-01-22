import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def _addLabel(self, text, label, g, rowNo, colNo, x, y, width, height, calcOnly=False):
    if label.visible:
        labelWidth = stringWidth(text, label.fontName, label.fontSize)
        flipXY = self._flipXY
        if flipXY:
            y0, x0, pm = self._labelXY(label, y, x, height, width)
        else:
            x0, y0, pm = self._labelXY(label, x, y, width, height)
        fixedEnd = getattr(label, 'fixedEnd', None)
        if fixedEnd is not None:
            v = fixedEnd._getValue(self, pm)
            x00, y00 = (x0, y0)
            if flipXY:
                x0 = v
            else:
                y0 = v
        elif flipXY:
            x00 = x0
            y00 = y + height / 2.0
        else:
            x00 = x + width / 2.0
            y00 = y0
        fixedStart = getattr(label, 'fixedStart', None)
        if fixedStart is not None:
            v = fixedStart._getValue(self, pm)
            if flipXY:
                x00 = v
            else:
                y00 = v
        if pm < 0:
            if flipXY:
                dx = -2 * label.dx
                dy = 0
            else:
                dy = -2 * label.dy
                dx = 0
        else:
            dy = dx = 0
        if calcOnly:
            return (x0 + dx, y0 + dy)
        label.setOrigin(x0 + dx, y0 + dy)
        label.setText(text)
        sC, sW = (label.lineStrokeColor, label.lineStrokeWidth)
        if sC and sW:
            g.insert(0, Line(x00, y00, x0, y0, strokeColor=sC, strokeWidth=sW))
        g.add(label)
        alx = getattr(self, 'barLabelCallOut', None)
        if alx:
            label._callOutInfo = (self, g, rowNo, colNo, x, y, width, height, x00, y00, x0, y0)
            alx(label)
            del label._callOutInfo
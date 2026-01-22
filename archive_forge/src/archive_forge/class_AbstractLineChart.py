from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, \
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, Polygon, PolyLine
from reportlab.graphics.widgets.signsandsymbols import NoEntry
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol, makeMarker
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from .utils import FillPairedData
class AbstractLineChart(PlotArea):

    def makeSwatchSample(self, rowNo, x, y, width, height):
        baseStyle = self.lines
        styleIdx = rowNo % len(baseStyle)
        style = baseStyle[styleIdx]
        color = style.strokeColor
        yh2 = y + height / 2.0
        lineStyle = getattr(style, 'lineStyle', None)
        if lineStyle == 'bar':
            dash = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
            strokeWidth = getattr(style, 'strokeWidth', getattr(style, 'strokeWidth', None))
            L = Rect(x, y, width, height, strokeWidth=strokeWidth, strokeColor=color, strokeLineCap=0, strokeDashArray=dash, fillColor=getattr(style, 'fillColor', color))
        elif self.joinedLines or lineStyle == 'joinedLine':
            dash = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
            strokeWidth = getattr(style, 'strokeWidth', getattr(style, 'strokeWidth', None))
            L = Line(x, yh2, x + width, yh2, strokeColor=color, strokeLineCap=0)
            if strokeWidth:
                L.strokeWidth = strokeWidth
            if dash:
                L.strokeDashArray = dash
        else:
            L = None
        if hasattr(style, 'symbol'):
            S = style.symbol
        elif hasattr(baseStyle, 'symbol'):
            S = baseStyle.symbol
        else:
            S = None
        if S:
            S = uSymbol2Symbol(S, x + width / 2.0, yh2, color)
        if S and L:
            g = Group()
            g.add(L)
            g.add(S)
            return g
        return S or L

    def getSeriesName(self, i, default=None):
        """return series name i or default"""
        return _objStr(getattr(self.lines[i], 'name', default))
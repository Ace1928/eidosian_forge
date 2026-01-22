from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Line, Rect, Polygon, PolyLine, \
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.piecharts import WedgeLabel
from reportlab.graphics.widgets.markers import makeMarker, uSymbol2Symbol, isSymbol
class StrandLabel(SpokeLabel):
    _attrMap = AttrMap(BASE=SpokeLabel, format=AttrMapValue(EitherOr((isStringOrNone, isCallable)), desc='Format for the label'), dR=AttrMapValue(isNumberOrNone, desc='radial shift for label'))

    def __init__(self, **kw):
        self.format = ''
        self.dR = 0
        SpokeLabel.__init__(self, **kw)
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
class SpokeProperty(PropHolder):
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='width'), fillColor=AttrMapValue(isColorOrNone, desc='filling color'), strokeColor=AttrMapValue(isColorOrNone, desc='stroke color'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='dashing pattern, e.g. (2,1)'), labelRadius=AttrMapValue(isNumber, desc='label radius', advancedUsage=1), visible=AttrMapValue(isBoolean, desc='True if the spoke line is to be drawn'))

    def __init__(self, **kw):
        self.strokeWidth = 0.5
        self.fillColor = None
        self.strokeColor = STATE_DEFAULTS['strokeColor']
        self.strokeDashArray = STATE_DEFAULTS['strokeDashArray']
        self.visible = 1
        self.labelRadius = 1.05
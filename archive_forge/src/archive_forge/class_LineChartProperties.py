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
class LineChartProperties(PropHolder):
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='Width of a line.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color of a line or border.'), fillColor=AttrMapValue(isColorOrNone, desc='fill color of a bar.'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array of a line.'), symbol=AttrMapValue(NoneOr(isSymbol), desc='Widget placed at data points.', advancedUsage=1), shader=AttrMapValue(None, desc='Shader Class.', advancedUsage=1), filler=AttrMapValue(None, desc='Filler Class.', advancedUsage=1), name=AttrMapValue(isStringOrNone, desc='Name of the line.'), lineStyle=AttrMapValue(NoneOr(OneOf('line', 'joinedLine', 'bar')), desc='What kind of plot this line is', advancedUsage=1), barWidth=AttrMapValue(isNumberOrNone, desc='Percentage of available width to be used for a bar', advancedUsage=1), inFill=AttrMapValue(isBoolean, desc='If true flood fill to x axis', advancedUsage=1))
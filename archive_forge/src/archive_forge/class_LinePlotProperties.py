from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten, isStr
from reportlab.graphics.shapes import Drawing, Group, Rect, PolyLine, Polygon, _SetKeyWordArgs
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from reportlab.graphics.charts.utils import *
from reportlab.graphics.widgets.markers import uSymbol2Symbol, makeMarker
from reportlab.graphics.widgets.grids import Grid, DoubleGrid, ShadedPolygon
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.charts.areas import PlotArea
from .utils import FillPairedData
from reportlab.graphics.charts.linecharts import AbstractLineChart
class LinePlotProperties(PropHolder):
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='Width of a line.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color of a line.'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array of a line.'), fillColor=AttrMapValue(isColorOrNone, desc='Color of infill defaults to the strokeColor.'), symbol=AttrMapValue(None, desc='Widget placed at data points.', advancedUsage=1), shader=AttrMapValue(None, desc='Shader Class.', advancedUsage=1), filler=AttrMapValue(None, desc='Filler Class.', advancedUsage=1), name=AttrMapValue(isStringOrNone, desc='Name of the line.'), lineStyle=AttrMapValue(NoneOr(OneOf('line', 'joinedLine', 'bar')), desc='What kind of plot this line is', advancedUsage=1), barWidth=AttrMapValue(isNumberOrNone, desc='Percentage of available width to be used for a bar', advancedUsage=1), inFill=AttrMapValue(isBoolean, desc='If true flood fill to x axis', advancedUsage=1))
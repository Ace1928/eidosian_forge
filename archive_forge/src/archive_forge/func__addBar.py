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
def _addBar(colNo, accx):
    if useAbsolute == 7:
        x = groupWidth * _cscale(colNo) + xVal + org
    else:
        g, _ = cScale(colNo)
        x = g + xVal
    datum = row[colNo]
    if datum is None:
        height = None
        y = baseLine
    else:
        if style not in ('parallel', 'parallel_3d') and (not isLine):
            if datum <= -1e-06:
                y = vScale(accumNeg[accx])
                if y < baseLine if vARD else y > baseLine:
                    y = baseLine
                accumNeg[accx] += datum
                datum = accumNeg[accx]
            else:
                y = vScale(accumPos[accx])
                if y > baseLine if vARD else y < baseLine:
                    y = baseLine
                accumPos[accx] += datum
                datum = accumPos[accx]
        else:
            y = baseLine
        height = vScale(datum) - y
        if -1e-08 < height <= 1e-08:
            height = 1e-08
            if datum < -1e-08:
                height = -1e-08
    barRow.append(flipXY and (y, x, height, width) or (x, y, width, height))
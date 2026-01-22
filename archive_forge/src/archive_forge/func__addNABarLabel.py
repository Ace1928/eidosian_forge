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
def _addNABarLabel(self, g, rowNo, colNo, x, y, width, height, calcOnly=False, na=None):
    if na is None:
        na = self.naLabel
    if na and na.text:
        na = copy.copy(na)
        v = self.valueAxis._valueMax <= 0 and -1e-08 or 1e-08
        if width is None:
            width = v
        if height is None:
            height = v
        return self._addLabel(na.text, na, g, rowNo, colNo, x, y, width, height, calcOnly=calcOnly)
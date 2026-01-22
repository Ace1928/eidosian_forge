from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
def _addPage(self, g, strokeWidth=3, color=None, dx=0, dy=0):
    x1, x2 = (31.85 + dx, 80.97 + dx)
    fL = 10
    y1, y2 = (dy - 34, dy + 50.5)
    L = [[x1, dy - 4, x1, y1, x2, y1, x2, dy - 1], [x1, dy + 11, x1, y2, x2 - fL, y2, x2, y2 - fL, x2, dy + 14], [x2 - 10, y2, x2 - 10, y2 - fL, x2, y2 - fL]]
    for l in L:
        g.add(PolyLine(l, strokeWidth=strokeWidth, strokeColor=color, strokeLineJoin=0))
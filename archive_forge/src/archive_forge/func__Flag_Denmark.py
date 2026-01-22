from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Denmark(self):
    s = _size
    g = Group()
    self._width = w = s * 1.4
    box = Rect(0, 0, w, s, fillColor=colors.red, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    whitebox1 = Rect(s / 5.0 * 2, 0, width=s / 6.0, height=s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whitebox1)
    whitebox2 = Rect(0, s / 2.0 - s / 12.0, width=w, height=s / 6.0, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whitebox2)
    return g
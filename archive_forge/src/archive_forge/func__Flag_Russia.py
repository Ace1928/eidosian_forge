from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Russia(self):
    s = _size
    g = Group()
    w = self._width = s * 1.5
    t = s / 3.0
    g.add(Rect(0, 0, width=w, height=t, fillColor=colors.red, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, t, width=w, height=t, fillColor=colors.blue, strokeColor=None, strokeWidth=0))
    g.add(Rect(0, 2 * t, width=w, height=t, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0))
    return g
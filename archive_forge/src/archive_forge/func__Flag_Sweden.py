from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Sweden(self):
    s = _size
    g = Group()
    self._width = s * 1.4
    box = Rect(0, 0, self._width, s, fillColor=colors.dodgerblue, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    box1 = Rect(s / 5.0 * 2, 0, width=s / 6.0, height=s, fillColor=colors.gold, strokeColor=None, strokeWidth=0)
    g.add(box1)
    box2 = Rect(0, s / 2.0 - s / 12.0, width=self._width, height=s / 6.0, fillColor=colors.gold, strokeColor=None, strokeWidth=0)
    g.add(box2)
    return g
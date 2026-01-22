from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Belgium(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.black, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    box1 = Rect(0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.black, strokeColor=None, strokeWidth=0)
    g.add(box1)
    box2 = Rect(s / 3.0 * 2.0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.gold, strokeColor=None, strokeWidth=0)
    g.add(box2)
    box3 = Rect(s / 3.0 * 4.0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.red, strokeColor=None, strokeWidth=0)
    g.add(box3)
    return g
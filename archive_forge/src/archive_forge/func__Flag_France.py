from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_France(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.navy, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    bluebox = Rect(0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.blue, strokeColor=None, strokeWidth=0)
    g.add(bluebox)
    whitebox = Rect(s / 3.0 * 2.0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.mintcream, strokeColor=None, strokeWidth=0)
    g.add(whitebox)
    redbox = Rect(s / 3.0 * 4.0, 0, width=s / 3.0 * 2.0, height=s, fillColor=colors.red, strokeColor=None, strokeWidth=0)
    g.add(redbox)
    return g
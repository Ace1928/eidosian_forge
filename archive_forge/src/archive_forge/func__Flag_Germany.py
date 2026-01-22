from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Germany(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.gold, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    blackbox1 = Rect(0, s / 3.0 * 2.0, width=s * 2.0, height=s / 3.0, fillColor=colors.black, strokeColor=None, strokeWidth=0)
    g.add(blackbox1)
    redbox1 = Rect(0, s / 3.0, width=s * 2.0, height=s / 3.0, fillColor=colors.orangered, strokeColor=None, strokeWidth=0)
    g.add(redbox1)
    return g
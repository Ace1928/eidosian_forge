from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_CzechRepublic(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.mintcream, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    redbox = Rect(0, 0, width=s * 2, height=s / 2.0, fillColor=colors.red, strokeColor=None, strokeWidth=0)
    g.add(redbox)
    bluewedge = Polygon(points=[0, 0, s, s / 2.0, 0, s], fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(bluewedge)
    return g
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_None(self):
    s = _size
    g = Group()
    g.add(Rect(0, 0, s * 2, s, fillColor=colors.purple, strokeColor=colors.black, strokeWidth=0))
    return g
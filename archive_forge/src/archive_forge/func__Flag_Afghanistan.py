from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Afghanistan(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.mintcream, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    greenbox = Rect(0, s / 3.0 * 2.0, width=s * 2.0, height=s / 3.0, fillColor=colors.limegreen, strokeColor=None, strokeWidth=0)
    g.add(greenbox)
    blackbox = Rect(0, 0, width=s * 2.0, height=s / 3.0, fillColor=colors.black, strokeColor=None, strokeWidth=0)
    g.add(blackbox)
    return g
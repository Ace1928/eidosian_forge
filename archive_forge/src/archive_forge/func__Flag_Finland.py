from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Finland(self):
    s = _size
    g = Group()
    box = Rect(0, 0, s * 2, s, fillColor=colors.ghostwhite, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    blueline1 = Rect(s * 0.6, 0, width=0.3 * s, height=s, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(blueline1)
    blueline2 = Rect(0, s * 0.4, width=s * 2, height=s * 0.3, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0)
    g.add(blueline2)
    return g
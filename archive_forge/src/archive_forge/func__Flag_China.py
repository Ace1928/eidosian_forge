from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_China(self):
    s = _size
    g = Group()
    self._width = w = s * 1.5
    g.add(Rect(0, 0, w, s, fillColor=colors.red, strokeColor=None, strokeWidth=0))

    def addStar(x, y, size, angle, g=g, w=s / 20.0, x0=0, y0=s / 2.0):
        s = Star()
        s.fillColor = colors.yellow
        s.angle = angle
        s.size = size * w * 2
        s.x = x * w + x0
        s.y = y * w + y0
        g.add(s)
    addStar(5, 5, 3, 0)
    addStar(10, 1, 1, 36.86989765)
    addStar(12, 3, 1, 8.213210702)
    addStar(12, 6, 1, 16.6015496)
    addStar(10, 8, 1, 53.13010235)
    return g
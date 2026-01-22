from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_EU(self):
    s = _size
    g = Group()
    w = self._width = 1.5 * s
    g.add(Rect(0, 0, w, s, fillColor=colors.darkblue, strokeColor=None, strokeWidth=0))
    centerx = w / 2.0
    centery = s / 2.0
    radius = s / 3.0
    yradius = radius
    xradius = radius
    nStars = 12
    delta = 2 * pi / nStars
    for i in range(nStars):
        rad = i * delta
        gs = Star()
        gs.x = cos(rad) * radius + centerx
        gs.y = sin(rad) * radius + centery
        gs.size = s / 10.0
        gs.fillColor = colors.gold
        g.add(gs)
    return g
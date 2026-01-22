from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Line, Rect, Polygon, Drawing, Group, String, Circle, Wedge
from reportlab.graphics import renderPDF
from reportlab.graphics.widgets.signsandsymbols import _Symbol
import copy
from math import sin, cos, pi
def _Flag_Cuba(self):
    s = _size
    g = Group()
    for i in range(5):
        stripe = Rect(0, i * s / 5.0, width=s * 2, height=s / 5.0, fillColor=[colors.darkblue, colors.mintcream][i % 2], strokeColor=None, strokeWidth=0)
        g.add(stripe)
    redwedge = Polygon(points=[0, 0, 4 * s / 5.0, s / 2.0, 0, s], fillColor=colors.red, strokeColor=None, strokeWidth=0)
    g.add(redwedge)
    star = Star()
    star.x = 2.5 * s / 10.0
    star.y = s / 2.0
    star.size = 3 * s / 10.0
    star.fillColor = colors.white
    g.add(star)
    box = Rect(0, 0, s * 2, s, fillColor=None, strokeColor=colors.black, strokeWidth=0)
    g.add(box)
    return g
import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
def theta0(data, direction):
    fac = 2 * pi / sum(data)
    rads = [d * fac for d in data]
    r0 = 0
    hrads = []
    for r in rads:
        hrads.append(r0 + r * 0.5)
        r0 += r
    vstar = len(data) * 1000000.0
    rstar = 0
    delta = pi / 36.0
    for i in range(36):
        r = i * delta
        v = sum([abs(sin(r + a)) for a in hrads])
        if v < vstar:
            if direction == 'clockwise':
                rstar = -r
            else:
                rstar = r
            vstar = v
    return rstar * 180 / pi
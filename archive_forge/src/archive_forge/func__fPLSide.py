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
def _fPLSide(l, width, side=None):
    data = l._origdata
    if side is None:
        li = data['li']
        ri = data['ri']
        if li is None:
            side = 1
            i = ri
        elif ri is None:
            side = 0
            i = li
        elif li[1] - li[0] > ri[1] - ri[0]:
            side = 0
            i = li
        else:
            side = 1
            i = ri
    w = data['width']
    edgePad = data['edgePad']
    if not side:
        l._pmv = 180
        l.x = edgePad + w
        i = data['li']
    else:
        l._pmv = 0
        l.x = width - w - edgePad
        i = data['ri']
    mid = data['mid'] = (i[0] + i[1]) * 0.5
    data['smid'] = sin(mid / _180_pi)
    data['cmid'] = cos(mid / _180_pi)
    data['side'] = side
    return (side, w)
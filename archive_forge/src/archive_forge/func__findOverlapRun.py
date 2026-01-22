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
def _findOverlapRun(B, i, wrap):
    """find overlap run containing B[i]"""
    n = len(B)
    R = [i]
    while 1:
        i = R[-1]
        j = (i + 1) % n
        if j in R or not boundsOverlap(B[i], B[j]):
            break
        R.append(j)
    while 1:
        i = R[0]
        j = (i - 1) % n
        if j in R or not boundsOverlap(B[i], B[j]):
            break
        R.insert(0, j)
    return R
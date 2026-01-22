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
class WedgeLabel(Label):

    def _checkDXY(self, ba):
        pass

    def _getBoxAnchor(self):
        ba = self.boxAnchor
        if ba in ('autox', 'autoy'):
            na = int(self._pmv % 360 / 45.0) * 45 % 360
            if not na % 90:
                da = (self._pmv - na) % 360
                if abs(da) > 5:
                    na += da > 0 and 45 or -45
            ba = (getattr(self, '_anti', None) and _ANGLE2RBOXANCHOR or _ANGLE2BOXANCHOR)[na]
            self._checkDXY(ba)
        return ba
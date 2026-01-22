from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _doPolygon(self, P):
    x, y = (self.x + self.dx, self.y + self.dy)
    if x or y:
        P = list(map(lambda i, P=P, A=[x, y]: P[i] + A[i & 1], list(range(len(P)))))
    return Polygon(P, strokeWidth=self.strokeWidth, strokeColor=self.strokeColor, fillColor=self.fillColor)
from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _Triangle(self):
    x, y = (self.x + self.dx, self.y + self.dy)
    r = float(self.size) / 2
    c = 30 * _toradians
    s = sin(30 * _toradians) * r
    c = cos(c) * r
    return self._doPolygon((0, r, -c, -s, c, -s))
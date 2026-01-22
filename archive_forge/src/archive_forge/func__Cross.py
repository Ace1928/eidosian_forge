from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def _Cross(self):
    x, y = (self.x + self.dx, self.y + self.dy)
    s = float(self.size)
    h, s = (s / 2, s / 6)
    return self._doPolygon((-s, -h, -s, -s, -h, -s, -h, s, -s, s, -s, h, s, h, s, s, h, s, h, -s, s, -s, s, -h))
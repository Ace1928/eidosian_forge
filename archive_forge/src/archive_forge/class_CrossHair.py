from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class CrossHair(_Symbol):
    """This draws an equilateral triangle."""
    _attrMap = AttrMap(BASE=_Symbol, innerGap=AttrMapValue(EitherOr((isString, isNumberOrNone)), desc=' gap at centre as "x%" or points or None'))

    def __init__(self):
        self.x = self.y = self.dx = self.dy = 0
        self.size = 10
        self.fillColor = None
        self.strokeColor = colors.black
        self.strokeWidth = 0.5
        self.innerGap = '20%'

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        ig = self.innerGap
        x = self.x + self.dx
        y = self.y + self.dy
        hsize = 0.5 * self.size
        if not ig:
            L = [(x - hsize, y, x + hsize, y), (x, y - hsize, x, y + hsize)]
        else:
            if isStr(ig):
                ig = asUnicode(ig)
                if ig.endswith(u'%'):
                    gs = hsize * float(ig[:-1]) / 100.0
                else:
                    gs = float(ig) * 0.5
            else:
                gs = ig * 0.5
            L = [(x - hsize, y, x - gs, y), (x + gs, y, x + hsize, y), (x, y - hsize, x, y - gs), (x, y + gs, x, y + hsize)]
        P = shapes.Path(strokeWidth=self.strokeWidth, strokeColor=self.strokeColor)
        for x0, y0, x1, y1 in L:
            P.moveTo(x0, y0)
            P.lineTo(x1, y1)
        g.add(P)
        return g
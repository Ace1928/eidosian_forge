from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class ETriangle(_Symbol):
    """This draws an equilateral triangle."""

    def __init__(self):
        _Symbol.__init__(self)

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        ae = s * 0.125
        triangle = shapes.Polygon(points=[self.x, self.y, self.x + s, self.y, self.x + s / 2, self.y + s], fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=s / 50.0)
        g.add(triangle)
        return g
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class DangerSign(_Symbol):
    """This draws a 'danger' sign: a yellow box with a black exclamation point.

        possible attributes:
        'x', 'y', 'size', 'strokeColor', 'fillColor', 'strokeWidth'

        """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.strokeColor = colors.black
        self.fillColor = colors.gold
        self.strokeWidth = self.size * 0.125

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        ew = self.strokeWidth
        ae = s * 0.125
        ew = self.strokeWidth
        ae = s * 0.125
        outerTriangle = shapes.Polygon(points=[self.x, self.y, self.x + s, self.y, self.x + s / 2, self.y + s], fillColor=None, strokeColor=self.strokeColor, strokeWidth=0)
        g.add(outerTriangle)
        innerTriangle = shapes.Polygon(points=[self.x + s / 50, self.y + s / 75, self.x + s - s / 50, self.y + s / 75, self.x + s / 2, self.y + s - s / 50], fillColor=self.fillColor, strokeColor=None, strokeWidth=0)
        g.add(innerTriangle)
        exmark = shapes.Polygon(points=[self.x + s / 2 - ew / 2, self.y + ae * 2.5, self.x + s / 2 + ew / 2, self.y + ae * 2.5, self.x + s / 2 + ew / 2 + ew / 6, self.y + ae * 5.5, self.x + s / 2 - ew / 2 - ew / 6, self.y + ae * 5.5], fillColor=self.strokeColor, strokeColor=None)
        g.add(exmark)
        exdot = shapes.Polygon(points=[self.x + s / 2 - ew / 2, self.y + ae, self.x + s / 2 + ew / 2, self.y + ae, self.x + s / 2 + ew / 2, self.y + ae * 2, self.x + s / 2 - ew / 2, self.y + ae * 2], fillColor=self.strokeColor, strokeColor=None)
        g.add(exdot)
        return g
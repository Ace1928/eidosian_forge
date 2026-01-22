from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class RTriangle(_Symbol):
    """This draws a right-angled triangle.

        possible attributes:
        'x', 'y', 'size', 'fillColor', 'strokeColor'

        """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.green
        self.strokeColor = None

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        ae = s * 0.125
        triangle = shapes.Polygon(points=[self.x, self.y, self.x + s, self.y, self.x, self.y + s], fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=s / 50.0)
        g.add(triangle)
        return g
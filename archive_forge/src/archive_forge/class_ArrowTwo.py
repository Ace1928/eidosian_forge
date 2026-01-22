from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class ArrowTwo(ArrowOne):
    """This widget draws an arrow (style two).

        possible attributes:
        'x', 'y', 'size', 'fillColor'

        """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.fillColor = colors.blue
        self.strokeWidth = 0
        self.strokeColor = None

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        x = self.x
        y = self.y
        s2 = s / 2
        s3 = s / 3
        s5 = s / 5
        s24 = s / 24
        g.add(shapes.Polygon(points=[x, y + 11 * s24, x, y + 13 * s24, x + 18.75 * s24, y + 13 * s24, x + 2 * s3, y + 2 * s3, x + s, y + s2, x + 2 * s3, y + s3, x + 18.75 * s24, y + 11 * s24], fillColor=self.fillColor, strokeColor=self.strokeColor, strokeWidth=self.strokeWidth))
        return g
from reportlab.lib import colors
from reportlab.lib.colors import black, white
from reportlab.graphics.shapes import Polygon, String, Drawing, Group, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.attrmap import *
from reportlab.lib.validators import *
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import getFont
from reportlab.graphics.widgets.grids import ShadedRect
def _getColors(self):
    numShades = self.numberOfBoxes + 1
    fillColorStart = self.startColor
    fillColorEnd = self.endColor
    colorsList = []
    for i in range(0, numShades):
        colorsList.append(colors.linearlyInterpolatedColor(fillColorStart, fillColorEnd, 0, numShades - 1, i))
    return colorsList
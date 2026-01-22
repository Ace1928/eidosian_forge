from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
class FloppyDisk(_Symbol):
    """This widget draws an icon of a floppy disk.

        possible attributes:
        'x', 'y', 'size', 'diskcolor'

        """
    _attrMap = AttrMap(BASE=_Symbol, diskColor=AttrMapValue(isColor))

    def __init__(self):
        self.x = 0
        self.y = 0
        self.size = 100
        self.diskColor = colors.black

    def draw(self):
        s = float(self.size)
        g = shapes.Group()
        diskBody = shapes.Rect(x=self.x, y=self.y + s / 100, width=s, height=s - s / 100, fillColor=self.diskColor, strokeColor=None, strokeWidth=0)
        g.add(diskBody)
        label = shapes.Rect(x=self.x + s * 0.1, y=self.y + s - s * 0.5, width=s * 0.8, height=s * 0.48, fillColor=colors.whitesmoke, strokeColor=None, strokeWidth=0)
        g.add(label)
        labelsplash = shapes.Rect(x=self.x + s * 0.1, y=self.y + s - s * 0.1, width=s * 0.8, height=s * 0.08, fillColor=colors.royalblue, strokeColor=None, strokeWidth=0)
        g.add(labelsplash)
        line1 = shapes.Line(x1=self.x + s * 0.15, y1=self.y + 0.6 * s, x2=self.x + s * 0.85, y2=self.y + 0.6 * s, fillColor=colors.black, strokeColor=colors.black, strokeWidth=0)
        g.add(line1)
        line2 = shapes.Line(x1=self.x + s * 0.15, y1=self.y + 0.7 * s, x2=self.x + s * 0.85, y2=self.y + 0.7 * s, fillColor=colors.black, strokeColor=colors.black, strokeWidth=0)
        g.add(line2)
        line3 = shapes.Line(x1=self.x + s * 0.15, y1=self.y + 0.8 * s, x2=self.x + s * 0.85, y2=self.y + 0.8 * s, fillColor=colors.black, strokeColor=colors.black, strokeWidth=0)
        g.add(line3)
        metalcover = shapes.Rect(x=self.x + s * 0.2, y=self.y, width=s * 0.5, height=s * 0.35, fillColor=colors.silver, strokeColor=None, strokeWidth=0)
        g.add(metalcover)
        coverslot = shapes.Rect(x=self.x + s * 0.28, y=self.y + s * 0.035, width=s * 0.12, height=s * 0.28, fillColor=self.diskColor, strokeColor=None, strokeWidth=0)
        g.add(coverslot)
        return g
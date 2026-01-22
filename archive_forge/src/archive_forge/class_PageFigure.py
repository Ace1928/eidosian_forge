import os
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import recursiveImport, strTypes
from reportlab.platypus import Frame
from reportlab.platypus import Flowable
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.validators import isColor
from reportlab.lib.colors import toColor
from reportlab.lib.styles import _baseFontName, _baseFontNameI
class PageFigure(Figure):
    """Shows a blank page in a frame, and draws on that.  Used in
    illustrations of how PLATYPUS works."""

    def __init__(self, background=None):
        Figure.__init__(self, 3 * inch, 3 * inch)
        self.caption = 'Figure 1 - a blank page'
        self.captionStyle = captionStyle
        self.background = background

    def drawVirtualPage(self):
        pass

    def drawFigure(self):
        drawPage(self.canv, 0.625 * inch, 0.25 * inch, 1.75 * inch, 2.5 * inch)
        self.canv.translate(0.625 * inch, 0.25 * inch)
        self.canv.scale(1.75 / 8.27, 2.5 / 11.69)
        self.drawVirtualPage()
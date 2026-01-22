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
class DrawingFigure(FlexFigure):
    """Drawing with a caption below it.  Clunky, scaling fails."""

    def __init__(self, modulename, classname, caption, baseDir=None, background=None):
        module = recursiveImport(modulename, baseDir)
        klass = getattr(module, classname)
        self.drawing = klass()
        FlexFigure.__init__(self, self.drawing.width, self.drawing.height, caption, background)
        self.growToFit = 1

    def drawFigure(self):
        self.canv.scale(self._scaleFactor, self._scaleFactor)
        self.drawing.drawOn(self.canv, 0, 0)
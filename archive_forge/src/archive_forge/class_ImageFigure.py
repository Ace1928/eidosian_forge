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
class ImageFigure(FlexFigure):
    """Image with a caption below it"""

    def __init__(self, filename, caption, background=None, scaleFactor=None, hAlign='CENTER', border=None):
        assert os.path.isfile(filename), 'image file %s not found' % filename
        from reportlab.lib.utils import ImageReader
        w, h = ImageReader(filename).getSize()
        self.filename = filename
        FlexFigure.__init__(self, w, h, caption, background, scaleFactor=scaleFactor, hAlign=hAlign, border=border)

    def drawFigure(self):
        self.canv.drawImage(self.filename, 0, 0, self.width, self.figureHeight)
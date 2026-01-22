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
def _getCaptionPara(self):
    caption = self.caption
    captionFont = self.captionFont
    captionSize = self.captionSize
    captionTextColor = self.captionTextColor
    captionBackColor = self.captionBackColor
    captionAlign = self.captionAlign
    captionPosition = self.captionPosition
    if self._captionData != (caption, captionFont, captionSize, captionTextColor, captionBackColor, captionAlign, captionPosition):
        self._captionData = (caption, captionFont, captionSize, captionTextColor, captionBackColor, captionAlign, captionPosition)
        if isinstance(caption, Paragraph):
            self.captionPara = caption
        elif isinstance(caption, strTypes):
            self.captionStyle = ParagraphStyle('Caption', fontName=captionFont, fontSize=captionSize, leading=1.2 * captionSize, textColor=captionTextColor, backColor=captionBackColor, spaceBefore=self.captionGap, alignment=TA_LEFT if captionAlign == 'left' else TA_RIGHT if captionAlign == 'right' else TA_CENTER)
            self.captionPara = Paragraph(self.caption, self.captionStyle)
        else:
            raise ValueError('Figure caption of type %r is not a string or Paragraph' % type(caption))
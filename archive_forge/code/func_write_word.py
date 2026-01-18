import io
import logging
import re
from typing import (
from pdfminer.pdfcolor import PDFColorSpace
from . import utils
from .image import ImageWriter
from .layout import LAParams, LTComponent, TextGroupElement
from .layout import LTAnno
from .layout import LTChar
from .layout import LTContainer
from .layout import LTCurve
from .layout import LTFigure
from .layout import LTImage
from .layout import LTItem
from .layout import LTLayoutContainer
from .layout import LTLine
from .layout import LTPage
from .layout import LTRect
from .layout import LTText
from .layout import LTTextBox
from .layout import LTTextBoxVertical
from .layout import LTTextGroup
from .layout import LTTextLine
from .pdfdevice import PDFTextDevice
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfinterp import PDFGraphicState, PDFResourceManager
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import AnyIO, Point, Matrix, Rect, PathSegment, make_compat_str
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import enc
from .utils import mult_matrix
def write_word(self) -> None:
    if len(self.working_text) > 0:
        bold_and_italic_styles = ''
        if 'Italic' in self.working_font:
            bold_and_italic_styles = 'font-style: italic; '
        if 'Bold' in self.working_font:
            bold_and_italic_styles += 'font-weight: bold; '
        self.write('<span style=\'font:"%s"; font-size:%d; %s\' class=\'ocrx_word\' title=\'%s; x_font %s; x_fsize %d\'>%s</span>' % (self.working_font, self.working_size, bold_and_italic_styles, self.bbox_repr(self.working_bbox), self.working_font, self.working_size, self.working_text.strip()))
    self.within_chars = False
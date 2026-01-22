import heapq
import logging
from typing import (
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdfinterp import Color
from .pdfinterp import PDFGraphicState
from .pdftypes import PDFStream
from .utils import INF, PathSegment
from .utils import LTComponentT
from .utils import Matrix
from .utils import Plane
from .utils import Point
from .utils import Rect
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import fsplit
from .utils import get_bound
from .utils import matrix2str
from .utils import uniq
class LTChar(LTComponent, LTText):
    """Actual letter in the text as a Unicode string."""

    def __init__(self, matrix: Matrix, font: PDFFont, fontsize: float, scaling: float, rise: float, text: str, textwidth: float, textdisp: Union[float, Tuple[Optional[float], float]], ncs: PDFColorSpace, graphicstate: PDFGraphicState) -> None:
        LTText.__init__(self)
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.ncs = ncs
        self.graphicstate = graphicstate
        self.adv = textwidth * fontsize * scaling
        if font.is_vertical():
            assert isinstance(textdisp, tuple)
            vx, vy = textdisp
            if vx is None:
                vx = fontsize * 0.5
            else:
                vx = vx * fontsize * 0.001
            vy = (1000 - vy) * fontsize * 0.001
            bbox_lower_left = (-vx, vy + rise + self.adv)
            bbox_upper_right = (-vx + fontsize, vy + rise)
        else:
            descent = font.get_descent() * fontsize
            bbox_lower_left = (0, descent + rise)
            bbox_upper_right = (self.adv, descent + rise + fontsize)
        a, b, c, d, e, f = self.matrix
        self.upright = 0 < a * d * scaling and b * c <= 0
        x0, y0 = apply_matrix_pt(self.matrix, bbox_lower_left)
        x1, y1 = apply_matrix_pt(self.matrix, bbox_upper_right)
        if x1 < x0:
            x0, x1 = (x1, x0)
        if y1 < y0:
            y0, y1 = (y1, y0)
        LTComponent.__init__(self, (x0, y0, x1, y1))
        if font.is_vertical():
            self.size = self.width
        else:
            self.size = self.height
        return

    def __repr__(self) -> str:
        return '<%s %s matrix=%s font=%r adv=%s text=%r>' % (self.__class__.__name__, bbox2str(self.bbox), matrix2str(self.matrix), self.fontname, self.adv, self.get_text())

    def get_text(self) -> str:
        return self._text

    def is_compatible(self, obj: object) -> bool:
        """Returns True if two characters can coexist in the same line."""
        return True
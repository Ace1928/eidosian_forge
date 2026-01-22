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
class LTPage(LTLayoutContainer):
    """Represents an entire page.

    Like any other LTLayoutContainer, an LTPage can be iterated to obtain child
    objects like LTTextBox, LTFigure, LTImage, LTRect, LTCurve and LTLine.
    """

    def __init__(self, pageid: int, bbox: Rect, rotate: float=0) -> None:
        LTLayoutContainer.__init__(self, bbox)
        self.pageid = pageid
        self.rotate = rotate
        return

    def __repr__(self) -> str:
        return '<%s(%r) %s rotate=%r>' % (self.__class__.__name__, self.pageid, bbox2str(self.bbox), self.rotate)
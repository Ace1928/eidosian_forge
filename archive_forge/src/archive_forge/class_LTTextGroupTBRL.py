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
class LTTextGroupTBRL(LTTextGroup):

    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        assert laparams.boxes_flow is not None
        boxes_flow = laparams.boxes_flow
        self._objs.sort(key=lambda obj: -(1 + boxes_flow) * (obj.x0 + obj.x1) - (1 - boxes_flow) * obj.y1)
        return
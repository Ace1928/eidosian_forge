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
class LTExpandableContainer(LTContainer[LTItemT]):

    def __init__(self) -> None:
        LTContainer.__init__(self, (+INF, +INF, -INF, -INF))
        return

    def add(self, obj: LTComponent) -> None:
        LTContainer.add(self, cast(LTItemT, obj))
        self.set_bbox((min(self.x0, obj.x0), min(self.y0, obj.y0), max(self.x1, obj.x1), max(self.y1, obj.y1)))
        return
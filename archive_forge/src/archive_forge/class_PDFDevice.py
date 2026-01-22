from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
class PDFDevice:
    """Translate the output of PDFPageInterpreter to the output that is needed"""

    def __init__(self, rsrcmgr: 'PDFResourceManager') -> None:
        self.rsrcmgr = rsrcmgr
        self.ctm: Optional[Matrix] = None

    def __repr__(self) -> str:
        return '<PDFDevice>'

    def __enter__(self) -> 'PDFDevice':
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def set_ctm(self, ctm: Matrix) -> None:
        self.ctm = ctm

    def begin_tag(self, tag: PSLiteral, props: Optional['PDFStackT']=None) -> None:
        pass

    def end_tag(self) -> None:
        pass

    def do_tag(self, tag: PSLiteral, props: Optional['PDFStackT']=None) -> None:
        pass

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        pass

    def end_page(self, page: PDFPage) -> None:
        pass

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        pass

    def end_figure(self, name: str) -> None:
        pass

    def paint_path(self, graphicstate: 'PDFGraphicState', stroke: bool, fill: bool, evenodd: bool, path: Sequence[PathSegment]) -> None:
        pass

    def render_image(self, name: str, stream: PDFStream) -> None:
        pass

    def render_string(self, textstate: 'PDFTextState', seq: PDFTextSeq, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> None:
        pass
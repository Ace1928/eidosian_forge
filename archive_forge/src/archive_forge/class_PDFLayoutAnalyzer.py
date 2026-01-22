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
class PDFLayoutAnalyzer(PDFTextDevice):
    cur_item: LTLayoutContainer
    ctm: Matrix

    def __init__(self, rsrcmgr: PDFResourceManager, pageno: int=1, laparams: Optional[LAParams]=None) -> None:
        PDFTextDevice.__init__(self, rsrcmgr)
        self.pageno = pageno
        self.laparams = laparams
        self._stack: List[LTLayoutContainer] = []

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        x0, y0, x1, y1 = page.mediabox
        x0, y0 = apply_matrix_pt(ctm, (x0, y0))
        x1, y1 = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(self.pageno, mediabox)

    def end_page(self, page: PDFPage) -> None:
        assert not self._stack, str(len(self._stack))
        assert isinstance(self.cur_item, LTPage), str(type(self.cur_item))
        if self.laparams is not None:
            self.cur_item.analyze(self.laparams)
        self.pageno += 1
        self.receive_layout(self.cur_item)

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))

    def end_figure(self, _: str) -> None:
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)

    def render_image(self, name: str, stream: PDFStream) -> None:
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        item = LTImage(name, stream, (self.cur_item.x0, self.cur_item.y0, self.cur_item.x1, self.cur_item.y1))
        self.cur_item.add(item)

    def paint_path(self, gstate: PDFGraphicState, stroke: bool, fill: bool, evenodd: bool, path: Sequence[PathSegment]) -> None:
        """Paint paths described in section 4.4 of the PDF reference manual"""
        shape = ''.join((x[0] for x in path))
        if shape[:1] != 'm':
            pass
        elif shape.count('m') > 1:
            for m in re.finditer('m[^m]+', shape):
                subpath = path[m.start(0):m.end(0)]
                self.paint_path(gstate, stroke, fill, evenodd, subpath)
        else:
            raw_pts = [cast(Point, p[-2:] if p[0] != 'h' else path[0][-2:]) for p in path]
            pts = [apply_matrix_pt(self.ctm, pt) for pt in raw_pts]
            operators = [str(operation[0]) for operation in path]
            transformed_points = [[apply_matrix_pt(self.ctm, (float(operand1), float(operand2))) for operand1, operand2 in zip(operation[1::2], operation[2::2])] for operation in path]
            transformed_path = [cast(PathSegment, (o, *p)) for o, p in zip(operators, transformed_points)]
            if shape in {'mlh', 'ml'}:
                line = LTLine(gstate.linewidth, pts[0], pts[1], stroke, fill, evenodd, gstate.scolor, gstate.ncolor, original_path=transformed_path, dashing_style=gstate.dash)
                self.cur_item.add(line)
            elif shape in {'mlllh', 'mllll'}:
                (x0, y0), (x1, y1), (x2, y2), (x3, y3), _ = pts
                is_closed_loop = pts[0] == pts[4]
                has_square_coordinates = x0 == x1 and y1 == y2 and (x2 == x3) and (y3 == y0) or (y0 == y1 and x1 == x2 and (y2 == y3) and (x3 == x0))
                if is_closed_loop and has_square_coordinates:
                    rect = LTRect(gstate.linewidth, (*pts[0], *pts[2]), stroke, fill, evenodd, gstate.scolor, gstate.ncolor, transformed_path, gstate.dash)
                    self.cur_item.add(rect)
                else:
                    curve = LTCurve(gstate.linewidth, pts, stroke, fill, evenodd, gstate.scolor, gstate.ncolor, transformed_path, gstate.dash)
                    self.cur_item.add(curve)
            else:
                curve = LTCurve(gstate.linewidth, pts, stroke, fill, evenodd, gstate.scolor, gstate.ncolor, transformed_path, gstate.dash)
                self.cur_item.add(curve)

    def render_char(self, matrix: Matrix, font: PDFFont, fontsize: float, scaling: float, rise: float, cid: int, ncs: PDFColorSpace, graphicstate: PDFGraphicState) -> float:
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(matrix, font, fontsize, scaling, rise, text, textwidth, textdisp, ncs, graphicstate)
        self.cur_item.add(item)
        return item.adv

    def handle_undefined_char(self, font: PDFFont, cid: int) -> str:
        log.debug('undefined: %r, %r', font, cid)
        return '(cid:%d)' % cid

    def receive_layout(self, ltpage: LTPage) -> None:
        pass
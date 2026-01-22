from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
class TextCanvas(Canvas):
    """
    class for storing rendered text and attributes
    """

    def __init__(self, text: Sequence[bytes] | None=None, attr: list[list[tuple[Hashable | None, int]]] | None=None, cs: list[list[tuple[Literal['0', 'U'] | None, int]]] | None=None, cursor: tuple[int, int] | None=None, maxcol: int | None=None, check_width: bool=True) -> None:
        """
        text -- list of strings, one for each line
        attr -- list of run length encoded attributes for text
        cs -- list of run length encoded character set for text
        cursor -- (x,y) of cursor or None
        maxcol -- screen columns taken by this canvas
        check_width -- check and fix width of all lines in text
        """
        super().__init__()
        if text is None:
            text = []
        if check_width:
            widths = []
            for t in text:
                if not isinstance(t, bytes):
                    raise CanvasError("Canvas text must be plain strings encoded in the screen's encoding", repr(text))
                widths.append(calc_width(t, 0, len(t)))
        else:
            if not isinstance(maxcol, int):
                raise TypeError(maxcol)
            widths = [maxcol] * len(text)
        if maxcol is None:
            if widths:
                maxcol = max(widths)
            else:
                maxcol = 0
        if attr is None:
            attr = [[] for _ in range(len(text))]
        if cs is None:
            cs = [[] for _ in range(len(text))]
        for i in range(len(text)):
            w = widths[i]
            if w > maxcol:
                raise CanvasError(f'Canvas text is wider than the maxcol specified:\nmaxcol={maxcol!r}\nwidths={widths!r}\ntext={text!r}\nurwid target encoding={get_encoding()}')
            if w < maxcol:
                text[i] += b''.rjust(maxcol - w)
            a_gap = len(text[i]) - rle_len(attr[i])
            if a_gap < 0:
                raise CanvasError(f'Attribute extends beyond text \n{text[i]!r}\n{attr[i]!r}')
            if a_gap:
                rle_append_modify(attr[i], (None, a_gap))
            cs_gap = len(text[i]) - rle_len(cs[i])
            if cs_gap < 0:
                raise CanvasError(f'Character Set extends beyond text \n{text[i]!r}\n{cs[i]!r}')
            if cs_gap:
                rle_append_modify(cs[i], (None, cs_gap))
        self._attr = attr
        self._cs = cs
        self.cursor = cursor
        self._text = text
        self._maxcol = maxcol

    def rows(self) -> int:
        """Return the number of rows in this canvas."""
        return len(self._text)

    def cols(self) -> int:
        """Return the screen column width of this canvas."""
        return self._maxcol

    def translated_coords(self, dx: int, dy: int) -> tuple[int, int] | None:
        """
        Return cursor coords shifted by (dx, dy), or None if there
        is no cursor.
        """
        if self.cursor:
            x, y = self.cursor
            return (x + dx, y + dy)
        return None

    def content(self, trim_left: int=0, trim_top: int=0, cols: int | None=0, rows: int | None=0, attr=None) -> Iterable[tuple[object, Literal['0', 'U'] | None, bytes]]:
        """
        Return the canvas content as a list of rows where each row
        is a list of (attr, cs, text) tuples.

        trim_left, trim_top, cols, rows may be set by
        CompositeCanvas when rendering a partially obscured
        canvas.
        """
        maxcol, maxrow = (self.cols(), self.rows())
        if not cols:
            cols = maxcol - trim_left
        if not rows:
            rows = maxrow - trim_top
        if not (0 <= trim_left < maxcol and (cols > 0 and trim_left + cols <= maxcol)):
            raise ValueError(trim_left)
        if not (0 <= trim_top < maxrow and (rows > 0 and trim_top + rows <= maxrow)):
            raise ValueError(trim_top)
        if trim_top or rows < maxrow:
            text_attr_cs = zip(self._text[trim_top:trim_top + rows], self._attr[trim_top:trim_top + rows], self._cs[trim_top:trim_top + rows])
        else:
            text_attr_cs = zip(self._text, self._attr, self._cs)
        for text, a_row, cs_row in text_attr_cs:
            if trim_left or cols < self._maxcol:
                text, a_row, cs_row = trim_text_attr_cs(text, a_row, cs_row, trim_left, trim_left + cols)
            attr_cs = rle_product(a_row, cs_row)
            i = 0
            row = []
            for (a, cs), run in attr_cs:
                if attr and a in attr:
                    a = attr[a]
                row.append((a, cs, text[i:i + run]))
                i += run
            yield row

    def content_delta(self, other: Canvas):
        """
        Return the differences between other and this canvas.

        If other is the same object as self this will return no
        differences, otherwise this is the same as calling
        content().
        """
        if other is self:
            return [self.cols()] * self.rows()
        return self.content()
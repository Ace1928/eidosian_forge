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
class SolidCanvas(Canvas):
    """
    A canvas filled completely with a single character.
    """

    def __init__(self, fill_char: str | bytes, cols: int, rows: int) -> None:
        super().__init__()
        end, col = calc_text_pos(fill_char, 0, len(fill_char), 1)
        if col != 1:
            raise ValueError(f'Invalid fill_char: {fill_char!r}')
        self._text, cs = apply_target_encoding(fill_char[:end])
        self._cs = cs[0][0]
        self.size = (cols, rows)
        self.cursor = None

    def cols(self) -> int:
        return self.size[0]

    def rows(self) -> int:
        return self.size[1]

    def content(self, trim_left: int=0, trim_top: int=0, cols: int | None=None, rows: int | None=None, attr=None) -> Iterable[list[tuple[object, Literal['0', 'U'] | None, bytes]]]:
        if cols is None:
            cols = self.size[0]
        if rows is None:
            rows = self.size[1]
        def_attr = None
        if attr and None in attr:
            def_attr = attr[None]
        line = [(def_attr, self._cs, self._text * cols)]
        for _ in range(rows):
            yield line

    def content_delta(self, other):
        """
        Return the differences between other and this canvas.
        """
        if other is self:
            return [self.cols()] * self.rows()
        return self.content()
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
class BlankCanvas(Canvas):
    """
    a canvas with nothing on it, only works as part of a composite canvas
    since it doesn't know its own size
    """

    def content(self, trim_left: int=0, trim_top: int=0, cols: int | None=0, rows: int | None=0, attr=None) -> Iterable[list[tuple[object, Literal['0', 'U'] | None, bytes]]]:
        """
        return (cols, rows) of spaces with default attributes.
        """
        def_attr = None
        if attr and None in attr:
            def_attr = attr[None]
        line = [(def_attr, None, b''.rjust(cols))]
        for _ in range(rows):
            yield line

    def cols(self) -> typing.NoReturn:
        raise NotImplementedError("BlankCanvas doesn't know its own size!")

    def rows(self) -> typing.NoReturn:
        raise NotImplementedError("BlankCanvas doesn't know its own size!")

    def content_delta(self, other: Canvas) -> typing.NoReturn:
        raise NotImplementedError("BlankCanvas doesn't know its own size!")
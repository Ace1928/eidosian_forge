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
class CompositeCanvas(Canvas):
    """
    class for storing a combination of canvases
    """

    def __init__(self, canv: Canvas=None) -> None:
        """
        canv -- a Canvas object to wrap this CompositeCanvas around.

        if canv is a CompositeCanvas, make a copy of its contents
        """
        super().__init__()
        if canv is None:
            self.shards: list[tuple[int, list[tuple[int, int, int, int, dict[Hashable | None, Hashable] | None, Canvas]]]] = []
            self.children: list[tuple[int, int, Canvas, typing.Any]] = []
        else:
            if hasattr(canv, 'shards'):
                self.shards = canv.shards
            else:
                self.shards = [(canv.rows(), [(0, 0, canv.cols(), canv.rows(), None, canv)])]
            self.children = [(0, 0, canv, None)]
            self.coords.update(canv.coords)
            for shortcut in canv.shortcuts:
                self.shortcuts[shortcut] = 'wrap'

    def __repr__(self) -> str:
        extra = ['']
        with contextlib.suppress(BaseException):
            extra.append(f'cols={self.cols()}')
        with contextlib.suppress(BaseException):
            extra.append(f'rows={self.rows()}')
        if self.cursor:
            extra.append(f'cursor={self.cursor}')
        if self.children:
            extra.append(f'children=({', '.join((repr(canv) for _, _, canv, _ in self.children))})')
        return f'<{self.__class__.__name__} finalized={bool(self.widget_info)}{' '.join(extra)} at 0x{id(self):X}>'

    def rows(self) -> int:
        for r, cv in self.shards:
            if not isinstance(r, int):
                raise TypeError(r, cv)
        return sum((r for r, cv in self.shards))

    def cols(self) -> int:
        if not self.shards:
            return 0
        cols = sum((cv[2] for cv in self.shards[0][1]))
        if not isinstance(cols, int):
            raise TypeError(cols)
        return cols

    def content(self, trim_left: int=0, trim_top: int=0, cols: int | None=None, rows: int | None=None, attr=None) -> Iterable[list[tuple[object, Literal['0', 'U'] | None, bytes]]]:
        """
        Return the canvas content as a list of rows where each row
        is a list of (attr, cs, text) tuples.
        """
        shard_tail = []
        for num_rows, cviews in self.shards:
            sbody = shard_body(cviews, shard_tail)
            for _ in range(num_rows):
                yield shard_body_row(sbody)
            shard_tail = shard_body_tail(num_rows, sbody)

    def content_delta(self, other: Canvas):
        """
        Return the differences between other and this canvas.
        """
        if not hasattr(other, 'shards'):
            yield from self.content()
            return
        shard_tail = []
        for num_rows, cviews in shards_delta(self.shards, other.shards):
            sbody = shard_body(cviews, shard_tail)
            row = []
            for _ in range(num_rows):
                if len(row) != 1 or not isinstance(row[0], int):
                    row = shard_body_row(sbody)
                yield row
            shard_tail = shard_body_tail(num_rows, sbody)

    def trim(self, top: int, count: int | None=None) -> None:
        """Trim lines from the top and/or bottom of canvas.

        top -- number of lines to remove from top
        count -- number of lines to keep, or None for all the rest
        """
        if top < 0:
            raise ValueError(f'invalid trim amount {top:d}!')
        if top >= self.rows():
            raise ValueError(f'cannot trim {top:d} lines from {self.rows():d}!')
        if self.widget_info:
            raise self._finalized_error
        if top:
            self.shards = shards_trim_top(self.shards, top)
        if count == 0:
            self.shards = []
        elif count is not None:
            self.shards = shards_trim_rows(self.shards, count)
        self.coords = self.translate_coords(0, -top)

    def trim_end(self, end: int) -> None:
        """Trim lines from the bottom of the canvas.

        end -- number of lines to remove from the end
        """
        if end <= 0:
            raise ValueError(f'invalid trim amount {end:d}!')
        if end > self.rows():
            raise ValueError(f'cannot trim {end:d} lines from {self.rows():d}!')
        if self.widget_info:
            raise self._finalized_error
        self.shards = shards_trim_rows(self.shards, self.rows() - end)

    def pad_trim_left_right(self, left: int, right: int) -> None:
        """
        Pad or trim this canvas on the left and right

        values > 0 indicate screen columns to pad
        values < 0 indicate screen columns to trim
        """
        if self.widget_info:
            raise self._finalized_error
        shards = self.shards
        if left < 0 or right < 0:
            trim_left = max(0, -left)
            cols = self.cols() - trim_left - max(0, -right)
            shards = shards_trim_sides(shards, trim_left, cols)
        rows = self.rows()
        if left > 0 or right > 0:
            top_rows, top_cviews = shards[0]
            if left > 0:
                new_top_cviews = [(0, 0, left, rows, None, blank_canvas), *top_cviews]
            else:
                new_top_cviews = top_cviews.copy()
            if right > 0:
                new_top_cviews.append((0, 0, right, rows, None, blank_canvas))
            shards = [(top_rows, new_top_cviews)] + shards[1:]
        self.coords = self.translate_coords(left, 0)
        self.shards = shards

    def pad_trim_top_bottom(self, top: int, bottom: int) -> None:
        """
        Pad or trim this canvas on the top and bottom.
        """
        if self.widget_info:
            raise self._finalized_error
        orig_shards = self.shards
        if top < 0 or bottom < 0:
            trim_top = max(0, -top)
            rows = self.rows() - trim_top - max(0, -bottom)
            self.trim(trim_top, rows)
        cols = self.cols()
        if top > 0:
            self.shards = [(top, [(0, 0, cols, top, None, blank_canvas)]), *self.shards]
            self.coords = self.translate_coords(0, top)
        if bottom > 0:
            if orig_shards is self.shards:
                self.shards = self.shards[:]
            self.shards.append((bottom, [(0, 0, cols, bottom, None, blank_canvas)]))

    def overlay(self, other: CompositeCanvas, left: int, top: int) -> None:
        """Overlay other onto this canvas."""
        if self.widget_info:
            raise self._finalized_error
        width = other.cols()
        height = other.rows()
        right = self.cols() - left - width
        bottom = self.rows() - top - height
        if right < 0:
            raise ValueError(f'top canvas of overlay not the size expected!{(other.cols(), left, right, width)!r}')
        if bottom < 0:
            raise ValueError(f'top canvas of overlay not the size expected!{(other.rows(), top, bottom, height)!r}')
        shards = self.shards
        top_shards = []
        side_shards = self.shards
        bottom_shards = []
        if top:
            side_shards = shards_trim_top(shards, top)
            top_shards = shards_trim_rows(shards, top)
        if bottom:
            bottom_shards = shards_trim_top(side_shards, height)
            side_shards = shards_trim_rows(side_shards, height)
        left_shards = []
        right_shards = []
        if left > 0:
            left_shards = [shards_trim_sides(side_shards, 0, left)]
        if right > 0:
            right_shards = [shards_trim_sides(side_shards, max(0, left + width), right)]
        if not self.rows():
            middle_shards = []
        elif left or right:
            middle_shards = shards_join((*left_shards, other.shards, *right_shards))
        else:
            middle_shards = other.shards
        self.shards = top_shards + middle_shards + bottom_shards
        self.coords.update(other.translate_coords(left, top))

    def fill_attr(self, a: Hashable) -> None:
        """
        Apply attribute a to all areas of this canvas with default attribute currently set to None,
        leaving other attributes intact.
        """
        self.fill_attr_apply({None: a})

    def fill_attr_apply(self, mapping: dict[Hashable | None, Hashable]) -> None:
        """
        Apply an attribute-mapping dictionary to the canvas.

        mapping -- dictionary of original-attribute:new-attribute items
        """
        if self.widget_info:
            raise self._finalized_error
        shards = []
        for num_rows, original_cviews in self.shards:
            new_cviews = []
            for cv in original_cviews:
                if cv[4] is None:
                    new_cviews.append(cv[:4] + (mapping,) + cv[5:])
                else:
                    combined = mapping.copy()
                    combined.update([(k, mapping.get(v, v)) for k, v in cv[4].items()])
                    new_cviews.append(cv[:4] + (combined,) + cv[5:])
            shards.append((num_rows, new_cviews))
        self.shards = shards

    def set_depends(self, widget_list: Sequence[Widget]) -> None:
        """
        Explicitly specify the list of widgets that this canvas
        depends on.  If any of these widgets change this canvas
        will have to be updated.
        """
        if self.widget_info:
            raise self._finalized_error
        self.depends_on = widget_list
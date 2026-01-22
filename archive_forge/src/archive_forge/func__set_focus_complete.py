from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def _set_focus_complete(self, size: tuple[int, int], focus: bool) -> None:
    """Finish setting the position now that we have maxcol & maxrow."""
    maxcol, maxrow = size
    self._invalidate()
    if self.set_focus_pending == 'first selectable':
        return self._set_focus_first_selectable((maxcol, maxrow), focus)
    if self.set_focus_valign_pending is not None:
        return self._set_focus_valign_complete((maxcol, maxrow), focus)
    coming_from, _focus_widget, focus_pos = self.set_focus_pending
    self.set_focus_pending = None
    _new_focus_widget, position = self._body.get_focus()
    if focus_pos == position:
        return None
    self._body.set_focus(focus_pos)
    middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus)
    focus_offset, _focus_widget, focus_pos, focus_rows, _cursor = middle
    _trim_top, fill_above = top
    _trim_bottom, fill_below = bottom
    offset = focus_offset
    for _widget, pos, rows in fill_above:
        offset -= rows
        if pos == position:
            self.change_focus((maxcol, maxrow), pos, offset, 'below')
            return None
    offset = focus_offset + focus_rows
    for _widget, pos, rows in fill_below:
        if pos == position:
            self.change_focus((maxcol, maxrow), pos, offset, 'above')
            return None
        offset += rows
    self._body.set_focus(position)
    widget, position = self._body.get_focus()
    rows = widget.rows((maxcol,), focus)
    if coming_from == 'below':
        offset = 0
    elif coming_from == 'above':
        offset = maxrow - rows
    else:
        offset = (maxrow - rows) // 2
    self.shift_focus((maxcol, maxrow), offset)
    return None
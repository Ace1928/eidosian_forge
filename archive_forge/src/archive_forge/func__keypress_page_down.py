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
def _keypress_page_down(self, size: tuple[int, int]) -> bool | None:
    maxcol, maxrow = size
    middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
    if middle is None:
        return True
    row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
    _trim_bottom, fill_below = bottom
    bottom_edge = maxrow - row_offset
    if not focus_widget.selectable():
        scroll_from_row = bottom_edge
    elif cursor is not None:
        _x, y = cursor
        scroll_from_row = y + 1
    elif bottom_edge >= focus_rows:
        scroll_from_row = focus_rows
    else:
        scroll_from_row = bottom_edge
    snap_rows = bottom_edge - scroll_from_row
    row_offset = -scroll_from_row
    scroll_from_row = bottom_edge = None
    t = [(row_offset, focus_widget, focus_pos, focus_rows)]
    pos = focus_pos
    row_offset += focus_rows
    for widget, pos, rows in fill_below:
        t.append((row_offset, widget, pos, rows))
        row_offset += rows
    snap_region_start = len(t)
    while row_offset < maxrow + snap_rows:
        widget, pos = self._body.get_next(pos)
        if widget is None:
            break
        rows = widget.rows((maxcol,))
        t.append((row_offset, widget, pos, rows))
        row_offset += rows
        if row_offset < maxrow:
            snap_region_start += 1
    row_offset, _w, _p, rows = t[-1]
    if row_offset + rows < maxrow:
        adjust = maxrow - (row_offset + rows)
        t = [(ro + adjust, w, p, r) for ro, w, p, r in t]
    row_offset, _w, _p, rows = t[0]
    if row_offset + rows <= 0:
        del t[0]
        snap_region_start -= 1
    self.update_pref_col_from_focus((maxcol, maxrow))
    search_order = list(range(snap_region_start, len(t))) + list(range(snap_region_start - 1, -1, -1))
    bad_choices = []
    cut_off_selectable_chosen = 0
    for i in search_order:
        row_offset, widget, pos, rows = t[i]
        if not widget.selectable():
            continue
        if not rows:
            continue
        pref_row = min(maxrow - row_offset - 1, rows - 1)
        if row_offset >= maxrow:
            self.change_focus((maxcol, maxrow), pos, maxrow - 1, 'above', (self.pref_col, 0), snap_rows + maxrow - row_offset - 1)
        else:
            self.change_focus((maxcol, maxrow), pos, row_offset, 'above', (self.pref_col, pref_row), snap_rows)
        middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
        act_row_offset, _ign1, _ign2, _ign3, _ign4 = middle
        if act_row_offset < row_offset - snap_rows:
            bad_choices.append(i)
            continue
        if act_row_offset > row_offset:
            bad_choices.append(i)
            continue
        if act_row_offset + rows > maxrow:
            bad_choices.append(i)
            cut_off_selectable_chosen = 1
            continue
        return None
    if cut_off_selectable_chosen:
        return None
    good_choices = [j for j in search_order if j not in bad_choices]
    for i in good_choices + search_order:
        row_offset, widget, pos, rows = t[i]
        if pos == focus_pos:
            continue
        if not rows:
            continue
        if row_offset >= maxrow:
            snap_rows -= snap_rows + maxrow - row_offset - 1
            row_offset = maxrow - 1
        self.change_focus((maxcol, maxrow), pos, row_offset, 'above', None, snap_rows)
        return None
    self.shift_focus((maxcol, maxrow), max(1 - focus_rows, row_offset))
    middle, _top, bottom = self.calculate_visible((maxcol, maxrow), True)
    act_row_offset, _ign1, pos, _ign2, _ign3 = middle
    if act_row_offset <= row_offset:
        return None
    if not t:
        return None
    _ign1, _ign2, pos, _ign3 = t[-1]
    widget, pos = self._body.get_next(pos)
    if widget is None:
        return None
    rows = widget.rows((maxcol,), True)
    self.change_focus((maxcol, maxrow), pos, maxrow - 1, 'above', (self.pref_col, 0), 0)
    return None
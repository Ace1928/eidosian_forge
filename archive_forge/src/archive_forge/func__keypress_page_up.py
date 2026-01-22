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
def _keypress_page_up(self, size: tuple[int, int]) -> bool | None:
    maxcol, maxrow = size
    middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
    if middle is None:
        return True
    row_offset, focus_widget, focus_pos, focus_rows, cursor = middle
    _trim_top, fill_above = top
    topmost_visible = row_offset
    if not focus_widget.selectable():
        scroll_from_row = topmost_visible
    elif cursor is not None:
        _x, y = cursor
        scroll_from_row = -y
    elif row_offset >= 0:
        scroll_from_row = 0
    else:
        scroll_from_row = topmost_visible
    snap_rows = topmost_visible - scroll_from_row
    row_offset = scroll_from_row + maxrow
    scroll_from_row = topmost_visible = None
    t = [(row_offset, focus_widget, focus_pos, focus_rows)]
    pos = focus_pos
    for widget, pos, rows in fill_above:
        row_offset -= rows
        t.append((row_offset, widget, pos, rows))
    snap_region_start = len(t)
    while row_offset > -snap_rows:
        widget, pos = self._body.get_prev(pos)
        if widget is None:
            break
        rows = widget.rows((maxcol,))
        row_offset -= rows
        if row_offset > 0:
            snap_region_start += 1
        t.append((row_offset, widget, pos, rows))
    row_offset, _w, _p, _r = t[-1]
    if row_offset > 0:
        adjust = -row_offset
        t = [(ro + adjust, w, p, r) for ro, w, p, r in t]
    row_offset, _w, _p, _r = t[0]
    if row_offset >= maxrow:
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
        pref_row = max(0, -row_offset)
        if rows + row_offset <= 0:
            self.change_focus((maxcol, maxrow), pos, -(rows - 1), 'below', (self.pref_col, rows - 1), snap_rows - (-row_offset - (rows - 1)))
        else:
            self.change_focus((maxcol, maxrow), pos, row_offset, 'below', (self.pref_col, pref_row), snap_rows)
        if fill_above and self._body.get_prev(fill_above[-1][1]) == (None, None):
            pass
        middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
        act_row_offset, _ign1, _ign2, _ign3, _ign4 = middle
        if act_row_offset > row_offset + snap_rows:
            bad_choices.append(i)
            continue
        if act_row_offset < row_offset:
            bad_choices.append(i)
            continue
        if act_row_offset < 0:
            bad_choices.append(i)
            cut_off_selectable_chosen = 1
            continue
        return None
    if cut_off_selectable_chosen:
        return None
    if fill_above and focus_widget.selectable() and (self._body.get_prev(fill_above[-1][1]) == (None, None)):
        pass
    good_choices = [j for j in search_order if j not in bad_choices]
    for i in good_choices + search_order:
        row_offset, widget, pos, rows = t[i]
        if pos == focus_pos:
            continue
        if not rows:
            continue
        if rows + row_offset <= 0:
            snap_rows -= -row_offset - (rows - 1)
            row_offset = -(rows - 1)
        self.change_focus((maxcol, maxrow), pos, row_offset, 'below', None, snap_rows)
        return None
    self.shift_focus((maxcol, maxrow), min(maxrow - 1, row_offset))
    middle, top, _bottom = self.calculate_visible((maxcol, maxrow), True)
    act_row_offset, _ign1, pos, _ign2, _ign3 = middle
    if act_row_offset >= row_offset:
        return None
    if not t:
        return None
    _ign1, _ign2, pos, _ign3 = t[-1]
    widget, pos = self._body.get_prev(pos)
    if widget is None:
        return None
    rows = widget.rows((maxcol,), True)
    self.change_focus((maxcol, maxrow), pos, -(rows - 1), 'below', (self.pref_col, rows - 1), 0)
    return None
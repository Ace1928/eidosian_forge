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
def _set_focus_valign_complete(self, size: tuple[int, int], focus: bool) -> None:
    """Finish setting the offset and inset now that we have have a maxcol & maxrow."""
    maxcol, maxrow = size
    vt, va = self.set_focus_valign_pending
    self.set_focus_valign_pending = None
    self.set_focus_pending = None
    focus_widget, _focus_pos = self._body.get_focus()
    if focus_widget is None:
        return
    rows = focus_widget.rows((maxcol,), focus)
    rtop, _rbot = calculate_top_bottom_filler(maxrow, vt, va, WHSettings.GIVEN, rows, None, 0, 0)
    self.shift_focus((maxcol, maxrow), rtop)
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
@runtime_checkable
class ScrollSupportingBody(Protocol):
    """Protocol for ListWalkers."""

    def get_focus(self) -> tuple[Widget, _K]:
        ...

    def set_focus(self, position: _K) -> None:
        ...

    def get_next(self, position: _K) -> tuple[Widget, _K] | tuple[None, None]:
        ...

    def get_prev(self, position: _K) -> tuple[Widget, _K] | tuple[None, None]:
        ...
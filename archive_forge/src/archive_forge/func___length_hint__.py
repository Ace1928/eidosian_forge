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
@property
def __length_hint__(self) -> Callable[[], int]:
    if isinstance(self._body, (Sized, EstimatedSized)):
        return lambda: operator.length_hint(self._body)
    raise AttributeError(f'{self._body.__class__.__name__} is not Sized and do not implement "__length_hint__"')
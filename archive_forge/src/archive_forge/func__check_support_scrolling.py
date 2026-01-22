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
def _check_support_scrolling(self) -> None:
    from .treetools import TreeWalker
    if not isinstance(self._body, ScrollSupportingBody):
        raise ListBoxError(f'{self} body do not implement methods required for scrolling protocol')
    if not isinstance(self._body, (Sized, EstimatedSized, TreeWalker)):
        raise ListBoxError(f"{self} body is not a Sized, can not estimate it's size and not a TreeWalker.Scroll is not allowed due to risk of infinite cycle of widgets load.")
    if getattr(self._body, 'wrap_around', False):
        raise ListBoxError('Body is wrapped around. Scroll position calculation is undefined.')
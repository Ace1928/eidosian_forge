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
class ListWalker(metaclass=signals.MetaSignals):
    signals: typing.ClassVar[list[str]] = ['modified']

    def _modified(self) -> None:
        signals.emit_signal(self, 'modified')

    def get_focus(self):
        """
        This default implementation relies on a focus attribute and a
        __getitem__() method defined in a subclass.

        Override and don't call this method if these are not defined.
        """
        try:
            focus = self.focus
            return (self[focus], focus)
        except (IndexError, KeyError, TypeError):
            return (None, None)

    def get_next(self, position):
        """
        This default implementation relies on a next_position() method and a
        __getitem__() method defined in a subclass.

        Override and don't call this method if these are not defined.
        """
        try:
            position = self.next_position(position)
            return (self[position], position)
        except (IndexError, KeyError):
            return (None, None)

    def get_prev(self, position):
        """
        This default implementation relies on a prev_position() method and a
        __getitem__() method defined in a subclass.

        Override and don't call this method if these are not defined.
        """
        try:
            position = self.prev_position(position)
            return (self[position], position)
        except (IndexError, KeyError):
            return (None, None)
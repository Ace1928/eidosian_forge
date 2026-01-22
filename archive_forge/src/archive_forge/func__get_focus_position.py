from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def _get_focus_position(self) -> int | None:
    warnings.warn(f'method `{self.__class__.__name__}._get_focus_position` is deprecated, please use `{self.__class__.__name__}.focus_position` property', DeprecationWarning, stacklevel=3)
    if not self.contents:
        raise IndexError('No focus_position, Pile is empty')
    return self.contents.focus
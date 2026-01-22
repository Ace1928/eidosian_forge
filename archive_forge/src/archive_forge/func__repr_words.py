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
def _repr_words(self) -> list[str]:
    if len(self.contents) > 1:
        contents_string = f'({len(self.contents)} items)'
    elif self.contents:
        contents_string = '(1 item)'
    else:
        contents_string = '()'
    return [*super()._repr_words(), contents_string]
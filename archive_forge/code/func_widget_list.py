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
@widget_list.setter
def widget_list(self, widgets):
    focus_position = self.focus_position
    self.contents = [(new, options) for new, (w, options) in zip(widgets, chain(self.contents, repeat((None, (WHSettings.WEIGHT, 1)))))]
    if focus_position < len(widgets):
        self.focus_position = focus_position
from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
@widget.setter
def widget(self, widget: Widget) -> None:
    self._widget = widget
    if self.pop_ups:
        self._topmost_widget.original_widget = self._widget
    else:
        self._topmost_widget = self._widget
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
def _test_process_input(self):
    """
        >>> w = _refl("widget")
        >>> w.selectable_rval = True
        >>> scr = _refl("screen")
        >>> scr.get_cols_rows_rval = (10, 5)
        >>> ml = MainLoop(w, [], scr)
        >>> ml.process_input(['enter', ('mouse drag', 1, 14, 20)])
        screen.get_cols_rows()
        widget.selectable()
        widget.keypress((10, 5), 'enter')
        widget.mouse_event((10, 5), 'mouse drag', 1, 14, 20, focus=True)
        True
        """
from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _stop_mouse_restore_buffer(self) -> None:
    """Stop mouse tracking and restore the screen."""
    self._mouse_tracking(False)
    move_cursor = ''
    if self._alternate_buffer:
        move_cursor = escape.RESTORE_NORMAL_BUFFER
    elif self.maxrow is not None:
        move_cursor = escape.set_cursor_position(0, self.maxrow)
    self.write(self._attrspec_to_escape(AttrSpec('', '')) + escape.SI + move_cursor + escape.SHOW_CURSOR)
    self.flush()
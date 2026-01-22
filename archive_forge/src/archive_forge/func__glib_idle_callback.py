from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop
def _glib_idle_callback(self):
    for callback in self._idle_callbacks.values():
        callback()
    self._glib_idle_enabled = False
    return False
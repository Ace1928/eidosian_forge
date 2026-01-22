from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop
def _enable_glib_idle(self) -> None:
    if self._glib_idle_enabled:
        return
    GLib.idle_add(self._glib_idle_callback)
    self._glib_idle_enabled = True
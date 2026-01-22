import locale
import unicodedata
import urllib.parse
from ctypes import *
from functools import lru_cache
from typing import Optional
import pyglet
from pyglet.window import WindowException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor
from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import key
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.canvas.xlib import XlibCanvas
from pyglet.libs.x11 import xlib
from pyglet.libs.x11 import cursorfont
from pyglet.util import asbytes
def _sync_resize(self):
    if self._enable_xsync and self._current_sync_valid:
        if xsync.XSyncValueIsZero(self._current_sync_value):
            self._current_sync_valid = False
            return
        xsync.XSyncSetCounter(self._x_display, self._sync_counter, self._current_sync_value)
        self._current_sync_value = None
        self._current_sync_valid = False
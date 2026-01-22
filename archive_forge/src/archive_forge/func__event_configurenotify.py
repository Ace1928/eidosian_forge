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
@XlibEventHandler(xlib.ConfigureNotify)
def _event_configurenotify(self, ev):
    if self._enable_xsync and self._current_sync_value:
        self._current_sync_valid = True
    if self._fullscreen:
        return
    self.switch_to()
    w, h = (ev.xconfigure.width, ev.xconfigure.height)
    x, y = (ev.xconfigure.x, ev.xconfigure.y)
    if self._width != w or self._height != h:
        self._width = w
        self._height = h
        self._update_view_size()
        self.dispatch_event('on_resize', self._width, self._height)
    if self._x != x or self._y != y:
        self.dispatch_event('on_move', x, y)
        self._x = x
        self._y = y
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
def _event_drag_drop(self, ev):
    if self._xdnd_version > XDND_VERSION:
        return
    time = xlib.CurrentTime
    if self._xdnd_format:
        if self._xdnd_version >= 1:
            time = ev.xclient.data.l[2]
        xlib.XConvertSelection(self._x_display, self._xdnd_atoms['XdndSelection'], self._xdnd_format, self._xdnd_atoms['XdndSelection'], self._window, time)
        xlib.XFlush(self._x_display)
    elif self._xdnd_version >= 2:
        e = xlib.XEvent()
        e.xclient.type = xlib.ClientMessage
        e.xclient.message_type = self._xdnd_atoms['XdndFinished']
        e.xclient.display = cast(self._x_display, POINTER(xlib.Display))
        e.xclient.window = self._window
        e.xclient.format = 32
        e.xclient.data.l[0] = self._window
        e.xclient.data.l[1] = 0
        e.xclient.data.l[2] = None
        xlib.XSendEvent(self._x_display, self._xdnd_source, False, xlib.NoEventMask, byref(e))
        xlib.XFlush(self._x_display)
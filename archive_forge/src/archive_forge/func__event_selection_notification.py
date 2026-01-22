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
@XlibEventHandler(xlib.SelectionNotify)
def _event_selection_notification(self, ev):
    if ev.xselection.property != 0 and ev.xselection.selection == self._xdnd_atoms['XdndSelection']:
        if self._xdnd_format:
            data, count, _ = self.get_single_property(ev.xselection.requestor, ev.xselection.property, ev.xselection.target)
            buffer = create_string_buffer(count)
            memmove(buffer, data, count)
            formatted_paths = self.parse_filenames(buffer.value.decode())
            e = xlib.XEvent()
            e.xclient.type = xlib.ClientMessage
            e.xclient.message_type = self._xdnd_atoms['XdndFinished']
            e.xclient.display = cast(self._x_display, POINTER(xlib.Display))
            e.xclient.window = self._window
            e.xclient.format = 32
            e.xclient.data.l[0] = self._xdnd_source
            e.xclient.data.l[1] = 1
            e.xclient.data.l[2] = self._xdnd_atoms['XdndActionCopy']
            xlib.XSendEvent(self._x_display, self._get_root(), False, xlib.NoEventMask, byref(e))
            xlib.XFlush(self._x_display)
            xlib.XFree(data)
            self.dispatch_event('on_file_drop', self._xdnd_position[0], self._height - self._xdnd_position[1], formatted_paths)
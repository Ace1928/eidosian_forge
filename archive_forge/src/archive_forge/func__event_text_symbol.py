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
def _event_text_symbol(self, ev):
    text = None
    symbol = xlib.KeySym()
    buffer = create_string_buffer(128)
    count = xlib.XLookupString(ev.xkey, buffer, len(buffer) - 1, byref(symbol), None)
    filtered = xlib.XFilterEvent(ev, ev.xany.window)
    if ev.type == xlib.KeyPress and (not filtered):
        status = c_int()
        if _have_utf8:
            encoding = 'utf8'
            count = xlib.Xutf8LookupString(self._x_ic, ev.xkey, buffer, len(buffer) - 1, byref(symbol), byref(status))
            if status.value == xlib.XBufferOverflow:
                raise NotImplementedError('TODO: XIM buffer resize')
        else:
            encoding = 'ascii'
            count = xlib.XLookupString(ev.xkey, buffer, len(buffer) - 1, byref(symbol), None)
            if count:
                status.value = xlib.XLookupBoth
        if status.value & (xlib.XLookupChars | xlib.XLookupBoth):
            text = buffer.value[:count].decode(encoding)
        if text and unicodedata.category(text) == 'Cc' and (text != '\r'):
            text = None
    symbol = symbol.value
    if ev.xkey.keycode == 0 and (not filtered):
        symbol = None
    if symbol and symbol not in key._key_names and ev.xkey.keycode:
        try:
            symbol = ord(chr(symbol).lower())
        except ValueError:
            symbol = key.user_key(ev.xkey.keycode)
        else:
            if symbol not in key._key_names:
                symbol = key.user_key(ev.xkey.keycode)
    if filtered:
        return (None, symbol)
    return (text, symbol)
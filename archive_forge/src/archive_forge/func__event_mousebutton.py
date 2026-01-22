from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
def _event_mousebutton(self, ev, button, lParam):
    if ev == 'on_mouse_press':
        _user32.SetCapture(self._view_hwnd)
    else:
        _user32.ReleaseCapture()
    x, y = self._get_location(lParam)
    y = self._height - y
    self.dispatch_event(ev, x, y, button, self._get_modifiers())
    return 0
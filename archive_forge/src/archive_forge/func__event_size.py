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
@Win32EventHandler(WM_SIZE)
def _event_size(self, msg, wParam, lParam):
    if not self._dc:
        return None
    if wParam == SIZE_MINIMIZED:
        self._hidden = True
        self.dispatch_event('on_hide')
        return 0
    if self._hidden:
        self._hidden = False
        self.dispatch_event('on_show')
    w, h = self._get_location(lParam)
    if not self._fullscreen:
        self._width, self._height = (w, h)
    self._update_view_location(self._width, self._height)
    if self._exclusive_mouse:
        self._update_clipped_cursor()
    self.switch_to()
    self.dispatch_event('on_resize', self._width, self._height)
    return 0
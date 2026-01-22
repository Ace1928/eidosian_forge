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
@Win32EventHandler(WM_KILLFOCUS)
def _event_killfocus(self, msg, wParam, lParam):
    self.dispatch_event('on_deactivate')
    self._has_focus = False
    exclusive_keyboard = self._exclusive_keyboard
    exclusive_mouse = self._exclusive_mouse
    self.set_exclusive_keyboard(False)
    self.set_exclusive_mouse(False)
    for symbol in self._keyboard_state:
        self._keyboard_state[symbol] = False
    self._exclusive_keyboard = exclusive_keyboard
    self._exclusive_keyboard_focus = False
    self._exclusive_mouse = exclusive_mouse
    self._exclusive_mouse_focus = False
    return 0
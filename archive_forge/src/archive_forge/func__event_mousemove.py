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
@ViewEventHandler
@Win32EventHandler(WM_MOUSEMOVE)
def _event_mousemove(self, msg, wParam, lParam):
    if self._exclusive_mouse and self._has_focus:
        return 0
    x, y = self._get_location(lParam)
    y = self._height - y
    dx = x - self._mouse_x
    dy = y - self._mouse_y
    if not self._tracking:
        self._mouse_in_window = True
        self.set_mouse_platform_visible()
        self.dispatch_event('on_mouse_enter', x, y)
        self._tracking = True
        track = TRACKMOUSEEVENT()
        track.cbSize = sizeof(track)
        track.dwFlags = TME_LEAVE
        track.hwndTrack = self._view_hwnd
        _user32.TrackMouseEvent(byref(track))
    if self._mouse_x == x and self._mouse_y == y:
        return 0
    self._mouse_x = x
    self._mouse_y = y
    buttons = 0
    if wParam & MK_LBUTTON:
        buttons |= mouse.LEFT
    if wParam & MK_MBUTTON:
        buttons |= mouse.MIDDLE
    if wParam & MK_RBUTTON:
        buttons |= mouse.RIGHT
    if wParam & MK_XBUTTON1:
        buttons |= mouse.MOUSE4
    if wParam & MK_XBUTTON2:
        buttons |= mouse.MOUSE5
    if buttons:
        modifiers = self._get_modifiers()
        self.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)
    else:
        self.dispatch_event('on_mouse_motion', x, y, dx, dy)
    return 0
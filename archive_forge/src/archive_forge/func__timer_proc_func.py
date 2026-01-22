import ctypes
from .base import PlatformEventLoop
from pyglet.libs.win32 import _kernel32, _user32, types, constants
from pyglet.libs.win32.types import *
def _timer_proc_func(self, hwnd, msg, timer, t):
    if self._timer_func:
        self._timer_func()
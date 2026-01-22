from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _update_density_and_dpi(self):
    if platform == 'win':
        from ctypes import windll
        self._density = 1.0
        try:
            hwnd = windll.user32.GetActiveWindow()
            self.dpi = float(windll.user32.GetDpiForWindow(hwnd))
            self._density = self.dpi / 96
        except AttributeError:
            pass
    else:
        self._density = self._win._get_gl_size()[0] / self._size[0]
        if self._is_desktop:
            self.dpi = self._density * 96.0
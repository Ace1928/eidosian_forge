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
class _WindowsSysDPIWatch:
    hwnd = None
    new_windProc = None
    old_windProc = None
    window: WindowBase = None

    def __init__(self, window: WindowBase):
        self.window = window

    def start(self):
        from kivy.input.providers.wm_common import WNDPROC, SetWindowLong_WndProc_wrapper
        from ctypes import windll
        self.hwnd = windll.user32.GetActiveWindow()
        self.new_windProc = WNDPROC(self._wnd_proc)
        self.old_windProc = SetWindowLong_WndProc_wrapper(self.hwnd, self.new_windProc)

    def stop(self):
        from kivy.input.providers.wm_common import SetWindowLong_WndProc_wrapper
        if self.hwnd is None:
            return
        self.new_windProc = SetWindowLong_WndProc_wrapper(self.hwnd, self.old_windProc)
        self.hwnd = self.new_windProc = self.old_windProc = None

    def _wnd_proc(self, hwnd, msg, wParam, lParam):
        from kivy.input.providers.wm_common import WM_DPICHANGED, WM_NCCALCSIZE
        from ctypes import windll
        if msg == WM_DPICHANGED:

            def clock_callback(*args):
                if x_dpi != y_dpi:
                    raise ValueError('Can only handle DPI that are same for x and y')
                self.window.dpi = x_dpi
            x_dpi = wParam & 65535
            y_dpi = wParam >> 16
            Clock.schedule_once(clock_callback, -1)
        elif Config.getboolean('graphics', 'resizable') and msg == WM_NCCALCSIZE and self.window.custom_titlebar:
            return 0
        return windll.user32.CallWindowProcW(self.old_windProc, hwnd, msg, wParam, lParam)
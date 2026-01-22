import os
from kivy.input.providers.wm_common import RECT, PEN_OR_TOUCH_MASK, \
from kivy.input.motionevent import MotionEvent
def _pen_wndProc(self, hwnd, msg, wParam, lParam):
    if msg == WM_TABLET_QUERYSYSTEMGESTURE:
        return QUERYSYSTEMGESTURE_WNDPROC
    if self._is_pen_message(msg):
        self._pen_handler(msg, wParam, lParam)
        return 1
    else:
        return windll.user32.CallWindowProcW(self.old_windProc, hwnd, msg, wParam, lParam)
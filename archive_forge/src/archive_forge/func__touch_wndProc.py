import os
from kivy.input.providers.wm_common import WNDPROC, \
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def _touch_wndProc(self, hwnd, msg, wParam, lParam):
    done = False
    if msg == WM_TABLET_QUERYSYSTEMGESTURE:
        return QUERYSYSTEMGESTURE_WNDPROC
    if msg == WM_TOUCH:
        done = self._touch_handler(msg, wParam, lParam)
    if msg >= WM_MOUSEMOVE and msg <= WM_MOUSELAST:
        done = self._mouse_handler(msg, wParam, lParam)
    if not done:
        return windll.user32.CallWindowProcW(self.old_windProc, hwnd, msg, wParam, lParam)
    return 1
import os
from kivy.input.providers.wm_common import RECT, PEN_OR_TOUCH_MASK, \
from kivy.input.motionevent import MotionEvent
def _pen_handler(self, msg, wParam, lParam):
    if msg not in (WM_LBUTTONDOWN, WM_MOUSEMOVE, WM_LBUTTONUP):
        return
    windll.user32.GetClientRect(self.hwnd, byref(win_rect))
    x = c_int16(lParam & 65535).value / float(win_rect.w)
    y = c_int16(lParam >> 16).value / float(win_rect.h)
    y = abs(1.0 - y)
    if msg == WM_LBUTTONDOWN:
        self.pen_events.appendleft(('begin', x, y))
        self.pen_status = True
    if msg == WM_MOUSEMOVE and self.pen_status:
        self.pen_events.appendleft(('update', x, y))
    if msg == WM_LBUTTONUP:
        self.pen_events.appendleft(('end', x, y))
        self.pen_status = False
import os
from kivy.input.providers.wm_common import WNDPROC, \
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class WM_MotionEventProvider(MotionEventProvider):

    def start(self):
        global Window
        if not Window:
            from kivy.core.window import Window
        self.touch_events = deque()
        self.touches = {}
        self.uid = 0
        self.hwnd = windll.user32.GetActiveWindow()
        windll.user32.RegisterTouchWindow(self.hwnd, 1)
        self.new_windProc = WNDPROC(self._touch_wndProc)
        self.old_windProc = SetWindowLong_WndProc_wrapper(self.hwnd, self.new_windProc)

    def update(self, dispatch_fn):
        c_rect = RECT()
        windll.user32.GetClientRect(self.hwnd, byref(c_rect))
        pt = POINT(x=0, y=0)
        windll.user32.ClientToScreen(self.hwnd, byref(pt))
        x_offset, y_offset = (pt.x, pt.y)
        usable_w, usable_h = (float(c_rect.w), float(c_rect.h))
        while True:
            try:
                t = self.touch_events.pop()
            except:
                break
            x = (t.screen_x() - x_offset) / usable_w
            y = 1.0 - (t.screen_y() - y_offset) / usable_h
            if t.event_type == 'begin':
                self.uid += 1
                self.touches[t.id] = WM_MotionEvent(self.device, self.uid, [x, y, t.size()])
                dispatch_fn('begin', self.touches[t.id])
            if t.event_type == 'update' and t.id in self.touches:
                self.touches[t.id].move([x, y, t.size()])
                dispatch_fn('update', self.touches[t.id])
            if t.event_type == 'end' and t.id in self.touches:
                touch = self.touches[t.id]
                touch.move([x, y, t.size()])
                touch.update_time_end()
                dispatch_fn('end', touch)
                del self.touches[t.id]

    def stop(self):
        windll.user32.UnregisterTouchWindow(self.hwnd)
        self.new_windProc = SetWindowLong_WndProc_wrapper(self.hwnd, self.old_windProc)

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

    def _touch_handler(self, msg, wParam, lParam):
        touches = (TOUCHINPUT * wParam)()
        windll.user32.GetTouchInputInfo(HANDLE(lParam), wParam, touches, sizeof(TOUCHINPUT))
        for i in range(wParam):
            self.touch_events.appendleft(touches[i])
        windll.user32.CloseTouchInputHandle(HANDLE(lParam))
        return True

    def _mouse_handler(self, msg, wparam, lParam):
        info = windll.user32.GetMessageExtraInfo()
        if info & PEN_OR_TOUCH_MASK == PEN_OR_TOUCH_SIGNATURE:
            if info & PEN_EVENT_TOUCH_MASK:
                return True
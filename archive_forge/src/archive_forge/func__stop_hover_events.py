from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def _stop_hover_events(self):
    funbind = EventLoop.window.funbind
    funbind('mouse_pos', self.begin_or_update_hover_event)
    funbind('system_size', self.update_hover_event)
    funbind('on_cursor_enter', self.begin_hover_event)
    funbind('on_cursor_leave', self.end_hover_event)
    funbind('on_close', self.end_hover_event)
    funbind('on_rotate', self.update_hover_event)
from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def update_touch_graphics(self, win, *args):
    for touch in self.touches.values():
        touch.update_graphics(win)
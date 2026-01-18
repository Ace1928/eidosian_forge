from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def update_graphics(self, win, create=False):
    global Color, Ellipse
    de = self.ud.get('_drawelement', None)
    if de is None and create:
        if Color is None:
            from kivy.graphics import Color, Ellipse
        with win.canvas.after:
            de = (Color(0.8, 0.2, 0.2, 0.7), Ellipse(size=(20, 20), segments=15))
        self.ud._drawelement = de
    if de is not None:
        self.push()
        w, h = win._get_effective_size()
        self.scale_for_screen(w, h, rotation=win.rotation)
        de[1].pos = (self.x - 10, self.y - 10)
        self.pop()
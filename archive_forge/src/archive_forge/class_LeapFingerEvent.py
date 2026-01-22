from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
class LeapFingerEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ('pos', 'pos3d')

    def depack(self, args):
        super().depack(args)
        if args[0] is None:
            return
        x, y, z = args
        self.sx = normalize(x, -150, 150)
        self.sy = normalize(y, 40, 460)
        self.sz = normalize(z, -350, 350)
        self.z = z
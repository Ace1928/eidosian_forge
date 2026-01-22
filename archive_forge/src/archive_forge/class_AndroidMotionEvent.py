import os
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.shape import ShapeRect
from kivy.input.motionevent import MotionEvent
class AndroidMotionEvent(MotionEvent):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ['pos', 'pressure', 'shape']

    def depack(self, args):
        self.sx, self.sy, self.pressure, radius = args
        self.shape = ShapeRect()
        self.shape.width = radius
        self.shape.height = radius
        super().depack(args)
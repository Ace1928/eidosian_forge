import ctypes
import threading
import collections
import os
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class MacMotionEvent(MotionEvent):
    """MotionEvent representing a contact point on the touchpad. Supports pos
    and shape profiles.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('is_touch', True)
        kwargs.setdefault('type_id', 'touch')
        super().__init__(*args, **kwargs)
        self.profile = ('pos', 'shape')

    def depack(self, args):
        self.shape = ShapeRect()
        self.sx, self.sy = (args[0], args[1])
        self.shape.width = args[2]
        self.shape.height = args[2]
        super().depack(args)

    def __str__(self):
        return '<MacMotionEvent id=%d pos=(%f, %f) device=%s>' % (self.id, self.sx, self.sy, self.device)
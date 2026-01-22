from os.path import exists
from time import time
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty, BooleanProperty, StringProperty, \
from kivy.input.motionevent import MotionEvent
from kivy.base import EventLoop
from kivy.logger import Logger
from ast import literal_eval
from functools import partial
class RecorderMotionEvent(MotionEvent):

    def depack(self, args):
        for key, value in list(args.items()):
            setattr(self, key, value)
        super(RecorderMotionEvent, self).depack(args)
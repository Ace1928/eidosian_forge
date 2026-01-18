from os.path import exists
from time import time
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty, BooleanProperty, StringProperty, \
from kivy.input.motionevent import MotionEvent
from kivy.base import EventLoop
from kivy.logger import Logger
from ast import literal_eval
from functools import partial
Profile to save in the fake motion event when replayed.

    :attr:`record_profile_mask` is a :class:`~kivy.properties.ListProperty` and
    defaults to ['pos'].
    
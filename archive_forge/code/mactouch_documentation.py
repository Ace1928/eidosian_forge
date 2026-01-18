import ctypes
import threading
import collections
import os
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
MotionEvent representing a contact point on the touchpad. Supports pos
    and shape profiles.
    
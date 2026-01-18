from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
def tick_draw(self):
    """Tick the drawing counter.
        """
    self._process_events_before_frame()
    self._rfps_counter += 1
    self._frames_displayed += 1
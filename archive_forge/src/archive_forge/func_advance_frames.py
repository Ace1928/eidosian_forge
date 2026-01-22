import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def advance_frames(self, count):
    """Render the new frames and:

        * tick the Clock
        * dispatch input from all registered providers
        * flush all the canvas operations
        * redraw Window canvas if necessary
        """
    from kivy.base import EventLoop
    for i in range(count):
        EventLoop.idle()
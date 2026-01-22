import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
class RecordingListener(Listener):
    """A listener that records all events and stores them in the ``.buffer``
    attribute as a list of 2-tuple ``(float, Event)``, where the first element
    is the time the event occurred as returned by ``time.time()`` and the second
    element is the event.
    """

    def __init__(self):
        self.buffer = []

    def on_start(self, event):
        self.buffer.append((time.time(), event))

    def on_end(self, event):
        self.buffer.append((time.time(), event))
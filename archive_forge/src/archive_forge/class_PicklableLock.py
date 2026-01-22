from collections import defaultdict
from functools import partial
from threading import Lock
import inspect
import warnings
import logging
from transitions.core import Machine, Event, listify
class PicklableLock:
    """ A wrapper for threading.Lock which discards its state during pickling and
        is reinitialized unlocked when unpickled.
    """

    def __init__(self):
        self.lock = Lock()

    def __getstate__(self):
        return ''

    def __setstate__(self, value):
        return self.__init__()

    def __enter__(self):
        self.lock.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.__exit__(exc_type, exc_val, exc_tb)
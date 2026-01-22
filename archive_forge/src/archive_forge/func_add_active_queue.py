import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
@classmethod
def add_active_queue(cls, queue):
    """Makes a queue the currently active recording context."""
    cls._active_contexts.append(queue)
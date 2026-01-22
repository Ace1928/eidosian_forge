import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
class _PoolCache(dict):
    """
    Class that implements a cache for the Pool class that will notify
    the pool management threads every time the cache is emptied. The
    notification is done by the use of a queue that is provided when
    instantiating the cache.
    """

    def __init__(self, /, *args, notifier=None, **kwds):
        self.notifier = notifier
        super().__init__(*args, **kwds)

    def __delitem__(self, item):
        super().__delitem__(item)
        if not self:
            self.notifier.put(None)
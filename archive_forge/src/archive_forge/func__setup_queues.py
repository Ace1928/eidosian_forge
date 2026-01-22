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
def _setup_queues(self):
    self._inqueue = queue.SimpleQueue()
    self._outqueue = queue.SimpleQueue()
    self._quick_put = self._inqueue.put
    self._quick_get = self._outqueue.get
from collections import deque
from contextlib import contextmanager
import sys
import time
from eventlet.pools import Pool
from eventlet import timeout
from eventlet import hubs
from eventlet.hubs.timer import Timer
from eventlet.greenthread import GreenThread
def _proxy_method(self, *args, **kwargs):
    return getattr(self._base, _proxy_fun)(*args, **kwargs)
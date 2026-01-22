import collections
import contextlib
import threading
from fasteners import _utils
import six
@staticmethod
def _fetch_current_thread_functor():
    if eventlet is not None and eventlet_patcher is not None:
        if eventlet_patcher.is_monkey_patched('thread'):
            return eventlet.getcurrent
    return threading.current_thread
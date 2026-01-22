import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
class GreenFuture(Future):
    __doc__ = Future.__doc__

    def __init__(self):
        super(GreenFuture, self).__init__()
        if not _utils.EVENTLET_AVAILABLE:
            raise RuntimeError('Eventlet is needed to use a green future')
        if not _green.is_monkey_patched('thread'):
            self._condition = _green.threading.condition_object()
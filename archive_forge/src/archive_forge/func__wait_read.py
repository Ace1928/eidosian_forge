from __future__ import annotations
import sys
import time
import warnings
import gevent
from gevent.event import AsyncResult
from gevent.hub import get_hub
import zmq
from zmq import Context as _original_Context
from zmq import Socket as _original_Socket
from .poll import _Poller
def _wait_read(self):
    assert self.__readable.ready(), 'Only one greenlet can be waiting on this event'
    self.__readable = AsyncResult()
    tic = time.time()
    dt = self._gevent_bug_timeout
    if dt:
        timeout = gevent.Timeout(seconds=dt)
    else:
        timeout = None
    try:
        if timeout:
            timeout.start()
        self.__readable.get(block=True)
    except gevent.Timeout as t:
        if t is not timeout:
            raise
        toc = time.time()
        if self._debug_gevent and timeout and (toc - tic > dt) and self.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            print('BUG: gevent may have missed a libzmq recv event on %i!' % self.FD, file=sys.stderr)
    finally:
        if timeout:
            timeout.close()
        self.__readable.set()
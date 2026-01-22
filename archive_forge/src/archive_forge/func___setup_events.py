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
def __setup_events(self):
    self.__readable = AsyncResult()
    self.__writable = AsyncResult()
    self.__readable.set()
    self.__writable.set()
    try:
        self._state_event = get_hub().loop.io(self.getsockopt(zmq.FD), 1)
        self._state_event.start(self.__state_changed)
    except AttributeError:
        from gevent.core import read_event
        self._state_event = read_event(self.getsockopt(zmq.FD), self.__state_changed, persist=True)
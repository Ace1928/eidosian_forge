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
def __cleanup_events(self):
    if getattr(self, '_state_event', None):
        _stop(self._state_event)
        self._state_event = None
    self.__writable.set()
    self.__readable.set()
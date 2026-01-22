from __future__ import annotations
import asyncio
import pickle
import warnings
from queue import Queue
from typing import Any, Awaitable, Callable, Sequence, cast, overload
from tornado.ioloop import IOLoop
from tornado.log import gen_log
import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi
def _rebuild_io_state(self):
    """rebuild io state based on self.sending() and receiving()"""
    if self.socket is None:
        return
    state = 0
    if self.receiving():
        state |= zmq.POLLIN
    if self.sending():
        state |= zmq.POLLOUT
    self._state = state
    self._update_handler(state)
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
def _update_handler(self, state):
    """Update IOLoop handler with state."""
    if self.socket is None:
        return
    if state & self.socket.events:
        self.io_loop.add_callback(lambda: self._handle_events(self.socket, 0))
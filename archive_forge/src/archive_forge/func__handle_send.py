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
def _handle_send(self):
    """Handle a send event."""
    if self._flushed:
        return
    if not self.sending():
        gen_log.error("Shouldn't have handled a send event")
        return
    msg, kwargs = self._send_queue.get()
    try:
        status = self.socket.send_multipart(msg, **kwargs)
    except zmq.ZMQError as e:
        gen_log.error('SEND Error: %s', e)
        status = e
    if self._send_callback:
        callback = self._send_callback
        self._run_callback(callback, msg, status)
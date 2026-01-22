from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
def _unwatch_raw_sockets(self, loop, *sockets):
    """Unschedule callback for a raw socket"""
    for socket in sockets:
        loop.remove_handler(socket)
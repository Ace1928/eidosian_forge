from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
class _CancellableTornadoTimeout:

    def __init__(self, loop, timeout):
        self.loop = loop
        self.timeout = timeout

    def cancel(self):
        self.loop.remove_timeout(self.timeout)
import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _make_empty_waiter(self):
    if self._empty_waiter is not None:
        raise RuntimeError('Empty waiter is already set')
    self._empty_waiter = self._loop.create_future()
    if self._write_fut is None:
        self._empty_waiter.set_result(None)
    return self._empty_waiter
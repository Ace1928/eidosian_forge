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
def _call_connection_lost(self, exc):
    if self._called_connection_lost:
        return
    try:
        self._protocol.connection_lost(exc)
    finally:
        if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        self._sock = None
        server = self._server
        if server is not None:
            server._detach()
            self._server = None
        self._called_connection_lost = True
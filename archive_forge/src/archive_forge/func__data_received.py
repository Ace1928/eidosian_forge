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
def _data_received(self, data, length):
    if self._paused:
        assert self._pending_data_length == -1
        self._pending_data_length = length
        return
    if length == 0:
        self._eof_received()
        return
    if isinstance(self._protocol, protocols.BufferedProtocol):
        try:
            protocols._feed_data_to_buffered_proto(self._protocol, data)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal error: protocol.buffer_updated() call failed.')
            return
    else:
        self._protocol.data_received(data)
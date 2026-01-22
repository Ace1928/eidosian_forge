import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
class BaseQuicStream:

    def __init__(self, connection, stream_id):
        self._connection = connection
        self._stream_id = stream_id
        self._buffer = Buffer()
        self._expecting = 0

    def id(self):
        return self._stream_id

    def _expiration_from_timeout(self, timeout):
        if timeout is not None:
            expiration = time.time() + timeout
        else:
            expiration = None
        return expiration

    def _timeout_from_expiration(self, expiration):
        if expiration is not None:
            timeout = max(expiration - time.time(), 0.0)
        else:
            timeout = None
        return timeout

    def _encapsulate(self, datagram):
        l = len(datagram)
        return struct.pack('!H', l) + datagram

    def _common_add_input(self, data, is_end):
        self._buffer.put(data, is_end)
        try:
            return self._expecting > 0 and self._buffer.have(self._expecting)
        except UnexpectedEOF:
            return True

    def _close(self):
        self._connection.close_stream(self._stream_id)
        self._buffer.put(b'', True)
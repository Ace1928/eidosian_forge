import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
class BaseQuicConnection:

    def __init__(self, connection, address, port, source=None, source_port=0, manager=None):
        self._done = False
        self._connection = connection
        self._address = address
        self._port = port
        self._closed = False
        self._manager = manager
        self._streams = {}
        self._af = dns.inet.af_for_address(address)
        self._peer = dns.inet.low_level_address_tuple((address, port))
        if source is None and source_port != 0:
            if self._af == socket.AF_INET:
                source = '0.0.0.0'
            elif self._af == socket.AF_INET6:
                source = '::'
            else:
                raise NotImplementedError
        if source:
            self._source = (source, source_port)
        else:
            self._source = None

    def close_stream(self, stream_id):
        del self._streams[stream_id]

    def _get_timer_values(self, closed_is_special=True):
        now = time.time()
        expiration = self._connection.get_timer()
        if expiration is None:
            expiration = now + 3600
        interval = max(expiration - now, 0)
        if self._closed and closed_is_special:
            interval = min(interval, 0.05)
        return (expiration, interval)

    def _handle_timer(self, expiration):
        now = time.time()
        if expiration <= now:
            self._connection.handle_timer(now)
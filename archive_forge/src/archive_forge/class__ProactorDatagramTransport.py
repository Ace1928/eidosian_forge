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
class _ProactorDatagramTransport(_ProactorBasePipeTransport, transports.DatagramTransport):
    max_size = 256 * 1024

    def __init__(self, loop, sock, protocol, address=None, waiter=None, extra=None):
        self._address = address
        self._empty_waiter = None
        self._buffer_size = 0
        super().__init__(loop, sock, protocol, waiter=waiter, extra=extra)
        self._buffer = collections.deque()
        self._loop.call_soon(self._loop_reading)

    def _set_extra(self, sock):
        _set_socket_extra(self, sock)

    def get_write_buffer_size(self):
        return self._buffer_size

    def abort(self):
        self._force_close(None)

    def sendto(self, data, addr=None):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be bytes-like object (%r)', type(data))
        if not data:
            return
        if self._address is not None and addr not in (None, self._address):
            raise ValueError(f'Invalid address: must be None or {self._address}')
        if self._conn_lost and self._address:
            if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
                logger.warning('socket.sendto() raised exception.')
            self._conn_lost += 1
            return
        self._buffer.append((bytes(data), addr))
        self._buffer_size += len(data)
        if self._write_fut is None:
            self._loop_writing()
        self._maybe_pause_protocol()

    def _loop_writing(self, fut=None):
        try:
            if self._conn_lost:
                return
            assert fut is self._write_fut
            self._write_fut = None
            if fut:
                fut.result()
            if not self._buffer or (self._conn_lost and self._address):
                if self._closing:
                    self._loop.call_soon(self._call_connection_lost, None)
                return
            data, addr = self._buffer.popleft()
            self._buffer_size -= len(data)
            if self._address is not None:
                self._write_fut = self._loop._proactor.send(self._sock, data)
            else:
                self._write_fut = self._loop._proactor.sendto(self._sock, data, addr=addr)
        except OSError as exc:
            self._protocol.error_received(exc)
        except Exception as exc:
            self._fatal_error(exc, 'Fatal write error on datagram transport')
        else:
            self._write_fut.add_done_callback(self._loop_writing)
            self._maybe_resume_protocol()

    def _loop_reading(self, fut=None):
        data = None
        try:
            if self._conn_lost:
                return
            assert self._read_fut is fut or (self._read_fut is None and self._closing)
            self._read_fut = None
            if fut is not None:
                res = fut.result()
                if self._closing:
                    data = None
                    return
                if self._address is not None:
                    data, addr = (res, self._address)
                else:
                    data, addr = res
            if self._conn_lost:
                return
            if self._address is not None:
                self._read_fut = self._loop._proactor.recv(self._sock, self.max_size)
            else:
                self._read_fut = self._loop._proactor.recvfrom(self._sock, self.max_size)
        except OSError as exc:
            self._protocol.error_received(exc)
        except exceptions.CancelledError:
            if not self._closing:
                raise
        else:
            if self._read_fut is not None:
                self._read_fut.add_done_callback(self._loop_reading)
        finally:
            if data:
                self._protocol.datagram_received(data, addr)
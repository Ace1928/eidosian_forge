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
class _ProactorBasePipeTransport(transports._FlowControlMixin, transports.BaseTransport):
    """Base class for pipe and socket transports."""

    def __init__(self, loop, sock, protocol, waiter=None, extra=None, server=None):
        super().__init__(extra, loop)
        self._set_extra(sock)
        self._sock = sock
        self.set_protocol(protocol)
        self._server = server
        self._buffer = None
        self._read_fut = None
        self._write_fut = None
        self._pending_write = 0
        self._conn_lost = 0
        self._closing = False
        self._called_connection_lost = False
        self._eof_written = False
        if self._server is not None:
            self._server._attach()
        self._loop.call_soon(self._protocol.connection_made, self)
        if waiter is not None:
            self._loop.call_soon(futures._set_result_unless_cancelled, waiter, None)

    def __repr__(self):
        info = [self.__class__.__name__]
        if self._sock is None:
            info.append('closed')
        elif self._closing:
            info.append('closing')
        if self._sock is not None:
            info.append(f'fd={self._sock.fileno()}')
        if self._read_fut is not None:
            info.append(f'read={self._read_fut!r}')
        if self._write_fut is not None:
            info.append(f'write={self._write_fut!r}')
        if self._buffer:
            info.append(f'write_bufsize={len(self._buffer)}')
        if self._eof_written:
            info.append('EOF written')
        return '<{}>'.format(' '.join(info))

    def _set_extra(self, sock):
        self._extra['pipe'] = sock

    def set_protocol(self, protocol):
        self._protocol = protocol

    def get_protocol(self):
        return self._protocol

    def is_closing(self):
        return self._closing

    def close(self):
        if self._closing:
            return
        self._closing = True
        self._conn_lost += 1
        if not self._buffer and self._write_fut is None:
            self._loop.call_soon(self._call_connection_lost, None)
        if self._read_fut is not None:
            self._read_fut.cancel()
            self._read_fut = None

    def __del__(self, _warn=warnings.warn):
        if self._sock is not None:
            _warn(f'unclosed transport {self!r}', ResourceWarning, source=self)
            self._sock.close()

    def _fatal_error(self, exc, message='Fatal error on pipe transport'):
        try:
            if isinstance(exc, OSError):
                if self._loop.get_debug():
                    logger.debug('%r: %s', self, message, exc_info=True)
            else:
                self._loop.call_exception_handler({'message': message, 'exception': exc, 'transport': self, 'protocol': self._protocol})
        finally:
            self._force_close(exc)

    def _force_close(self, exc):
        if self._empty_waiter is not None and (not self._empty_waiter.done()):
            if exc is None:
                self._empty_waiter.set_result(None)
            else:
                self._empty_waiter.set_exception(exc)
        if self._closing and self._called_connection_lost:
            return
        self._closing = True
        self._conn_lost += 1
        if self._write_fut:
            self._write_fut.cancel()
            self._write_fut = None
        if self._read_fut:
            self._read_fut.cancel()
            self._read_fut = None
        self._pending_write = 0
        self._buffer = None
        self._loop.call_soon(self._call_connection_lost, exc)

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

    def get_write_buffer_size(self):
        size = self._pending_write
        if self._buffer is not None:
            size += len(self._buffer)
        return size
import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
class _SelectorSocketTransport(_SelectorTransport):
    _start_tls_compatible = True
    _sendfile_compatible = constants._SendfileMode.TRY_NATIVE

    def __init__(self, loop, sock, protocol, waiter=None, extra=None, server=None):
        self._read_ready_cb = None
        super().__init__(loop, sock, protocol, extra, server)
        self._eof = False
        self._empty_waiter = None
        base_events._set_nodelay(self._sock)
        self._loop.call_soon(self._protocol.connection_made, self)
        self._loop.call_soon(self._add_reader, self._sock_fd, self._read_ready)
        if waiter is not None:
            self._loop.call_soon(futures._set_result_unless_cancelled, waiter, None)

    def set_protocol(self, protocol):
        if isinstance(protocol, protocols.BufferedProtocol):
            self._read_ready_cb = self._read_ready__get_buffer
        else:
            self._read_ready_cb = self._read_ready__data_received
        super().set_protocol(protocol)

    def _read_ready(self):
        self._read_ready_cb()

    def _read_ready__get_buffer(self):
        if self._conn_lost:
            return
        try:
            buf = self._protocol.get_buffer(-1)
            if not len(buf):
                raise RuntimeError('get_buffer() returned an empty buffer')
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal error: protocol.get_buffer() call failed.')
            return
        try:
            nbytes = self._sock.recv_into(buf)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal read error on socket transport')
            return
        if not nbytes:
            self._read_ready__on_eof()
            return
        try:
            self._protocol.buffer_updated(nbytes)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal error: protocol.buffer_updated() call failed.')

    def _read_ready__data_received(self):
        if self._conn_lost:
            return
        try:
            data = self._sock.recv(self.max_size)
        except (BlockingIOError, InterruptedError):
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal read error on socket transport')
            return
        if not data:
            self._read_ready__on_eof()
            return
        try:
            self._protocol.data_received(data)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal error: protocol.data_received() call failed.')

    def _read_ready__on_eof(self):
        if self._loop.get_debug():
            logger.debug('%r received EOF', self)
        try:
            keep_open = self._protocol.eof_received()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._fatal_error(exc, 'Fatal error: protocol.eof_received() call failed.')
            return
        if keep_open:
            self._loop._remove_reader(self._sock_fd)
        else:
            self.close()

    def write(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f'data argument must be a bytes-like object, not {type(data).__name__!r}')
        if self._eof:
            raise RuntimeError('Cannot call write() after write_eof()')
        if self._empty_waiter is not None:
            raise RuntimeError('unable to write; sendfile is in progress')
        if not data:
            return
        if self._conn_lost:
            if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
                logger.warning('socket.send() raised exception.')
            self._conn_lost += 1
            return
        if not self._buffer:
            try:
                n = self._sock.send(data)
            except (BlockingIOError, InterruptedError):
                pass
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                self._fatal_error(exc, 'Fatal write error on socket transport')
                return
            else:
                data = data[n:]
                if not data:
                    return
            self._loop._add_writer(self._sock_fd, self._write_ready)
        self._buffer.extend(data)
        self._maybe_pause_protocol()

    def _write_ready(self):
        assert self._buffer, 'Data should not be empty'
        if self._conn_lost:
            return
        try:
            n = self._sock.send(self._buffer)
        except (BlockingIOError, InterruptedError):
            pass
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self._loop._remove_writer(self._sock_fd)
            self._buffer.clear()
            self._fatal_error(exc, 'Fatal write error on socket transport')
            if self._empty_waiter is not None:
                self._empty_waiter.set_exception(exc)
        else:
            if n:
                del self._buffer[:n]
            self._maybe_resume_protocol()
            if not self._buffer:
                self._loop._remove_writer(self._sock_fd)
                if self._empty_waiter is not None:
                    self._empty_waiter.set_result(None)
                if self._closing:
                    self._call_connection_lost(None)
                elif self._eof:
                    self._sock.shutdown(socket.SHUT_WR)

    def write_eof(self):
        if self._closing or self._eof:
            return
        self._eof = True
        if not self._buffer:
            self._sock.shutdown(socket.SHUT_WR)

    def can_write_eof(self):
        return True

    def _call_connection_lost(self, exc):
        super()._call_connection_lost(exc)
        if self._empty_waiter is not None:
            self._empty_waiter.set_exception(ConnectionError('Connection is closed by peer'))

    def _make_empty_waiter(self):
        if self._empty_waiter is not None:
            raise RuntimeError('Empty waiter is already set')
        self._empty_waiter = self._loop.create_future()
        if not self._buffer:
            self._empty_waiter.set_result(None)
        return self._empty_waiter

    def _reset_empty_waiter(self):
        self._empty_waiter = None
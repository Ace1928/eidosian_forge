import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
class BaseIOStream(object):
    """A utility class to write to and read from a non-blocking file or socket.

    We support a non-blocking ``write()`` and a family of ``read_*()``
    methods. When the operation completes, the ``Awaitable`` will resolve
    with the data read (or ``None`` for ``write()``). All outstanding
    ``Awaitables`` will resolve with a `StreamClosedError` when the
    stream is closed; `.BaseIOStream.set_close_callback` can also be used
    to be notified of a closed stream.

    When a stream is closed due to an error, the IOStream's ``error``
    attribute contains the exception object.

    Subclasses must implement `fileno`, `close_fd`, `write_to_fd`,
    `read_from_fd`, and optionally `get_fd_error`.

    """

    def __init__(self, max_buffer_size: Optional[int]=None, read_chunk_size: Optional[int]=None, max_write_buffer_size: Optional[int]=None) -> None:
        """`BaseIOStream` constructor.

        :arg max_buffer_size: Maximum amount of incoming data to buffer;
            defaults to 100MB.
        :arg read_chunk_size: Amount of data to read at one time from the
            underlying transport; defaults to 64KB.
        :arg max_write_buffer_size: Amount of outgoing data to buffer;
            defaults to unlimited.

        .. versionchanged:: 4.0
           Add the ``max_write_buffer_size`` parameter.  Changed default
           ``read_chunk_size`` to 64KB.
        .. versionchanged:: 5.0
           The ``io_loop`` argument (deprecated since version 4.1) has been
           removed.
        """
        self.io_loop = ioloop.IOLoop.current()
        self.max_buffer_size = max_buffer_size or 104857600
        self.read_chunk_size = min(read_chunk_size or 65536, self.max_buffer_size // 2)
        self.max_write_buffer_size = max_write_buffer_size
        self.error = None
        self._read_buffer = bytearray()
        self._read_buffer_size = 0
        self._user_read_buffer = False
        self._after_user_read_buffer = None
        self._write_buffer = _StreamBuffer()
        self._total_write_index = 0
        self._total_write_done_index = 0
        self._read_delimiter = None
        self._read_regex = None
        self._read_max_bytes = None
        self._read_bytes = None
        self._read_partial = False
        self._read_until_close = False
        self._read_future = None
        self._write_futures = collections.deque()
        self._close_callback = None
        self._connect_future = None
        self._ssl_connect_future = None
        self._connecting = False
        self._state = None
        self._closed = False

    def fileno(self) -> Union[int, ioloop._Selectable]:
        """Returns the file descriptor for this stream."""
        raise NotImplementedError()

    def close_fd(self) -> None:
        """Closes the file underlying this stream.

        ``close_fd`` is called by `BaseIOStream` and should not be called
        elsewhere; other users should call `close` instead.
        """
        raise NotImplementedError()

    def write_to_fd(self, data: memoryview) -> int:
        """Attempts to write ``data`` to the underlying file.

        Returns the number of bytes written.
        """
        raise NotImplementedError()

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        """Attempts to read from the underlying file.

        Reads up to ``len(buf)`` bytes, storing them in the buffer.
        Returns the number of bytes read. Returns None if there was
        nothing to read (the socket returned `~errno.EWOULDBLOCK` or
        equivalent), and zero on EOF.

        .. versionchanged:: 5.0

           Interface redesigned to take a buffer and return a number
           of bytes instead of a freshly-allocated object.
        """
        raise NotImplementedError()

    def get_fd_error(self) -> Optional[Exception]:
        """Returns information about any error on the underlying file.

        This method is called after the `.IOLoop` has signaled an error on the
        file descriptor, and should return an Exception (such as `socket.error`
        with additional information, or None if no such information is
        available.
        """
        return None

    def read_until_regex(self, regex: bytes, max_bytes: Optional[int]=None) -> Awaitable[bytes]:
        """Asynchronously read until we have matched the given regex.

        The result includes the data that matches the regex and anything
        that came before it.

        If ``max_bytes`` is not None, the connection will be closed
        if more than ``max_bytes`` bytes have been read and the regex is
        not satisfied.

        .. versionchanged:: 4.0
            Added the ``max_bytes`` argument.  The ``callback`` argument is
            now optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        future = self._start_read()
        self._read_regex = re.compile(regex)
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
            return future
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until(self, delimiter: bytes, max_bytes: Optional[int]=None) -> Awaitable[bytes]:
        """Asynchronously read until we have found the given delimiter.

        The result includes all the data read including the delimiter.

        If ``max_bytes`` is not None, the connection will be closed
        if more than ``max_bytes`` bytes have been read and the delimiter
        is not found.

        .. versionchanged:: 4.0
            Added the ``max_bytes`` argument.  The ``callback`` argument is
            now optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.
        """
        future = self._start_read()
        self._read_delimiter = delimiter
        self._read_max_bytes = max_bytes
        try:
            self._try_inline_read()
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
            return future
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_bytes(self, num_bytes: int, partial: bool=False) -> Awaitable[bytes]:
        """Asynchronously read a number of bytes.

        If ``partial`` is true, data is returned as soon as we have
        any bytes to return (but never more than ``num_bytes``)

        .. versionchanged:: 4.0
            Added the ``partial`` argument.  The callback argument is now
            optional and a `.Future` will be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` and ``streaming_callback`` arguments have
           been removed. Use the returned `.Future` (and
           ``partial=True`` for ``streaming_callback``) instead.

        """
        future = self._start_read()
        assert isinstance(num_bytes, numbers.Integral)
        self._read_bytes = num_bytes
        self._read_partial = partial
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_into(self, buf: bytearray, partial: bool=False) -> Awaitable[int]:
        """Asynchronously read a number of bytes.

        ``buf`` must be a writable buffer into which data will be read.

        If ``partial`` is true, the callback is run as soon as any bytes
        have been read.  Otherwise, it is run when the ``buf`` has been
        entirely filled with read data.

        .. versionadded:: 5.0

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        future = self._start_read()
        available_bytes = self._read_buffer_size
        n = len(buf)
        if available_bytes >= n:
            buf[:] = memoryview(self._read_buffer)[:n]
            del self._read_buffer[:n]
            self._after_user_read_buffer = self._read_buffer
        elif available_bytes > 0:
            buf[:available_bytes] = memoryview(self._read_buffer)[:]
        self._user_read_buffer = True
        self._read_buffer = buf
        self._read_buffer_size = available_bytes
        self._read_bytes = n
        self._read_partial = partial
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def read_until_close(self) -> Awaitable[bytes]:
        """Asynchronously reads all data from the socket until it is closed.

        This will buffer all available data until ``max_buffer_size``
        is reached. If flow control or cancellation are desired, use a
        loop with `read_bytes(partial=True) <.read_bytes>` instead.

        .. versionchanged:: 4.0
            The callback argument is now optional and a `.Future` will
            be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` and ``streaming_callback`` arguments have
           been removed. Use the returned `.Future` (and `read_bytes`
           with ``partial=True`` for ``streaming_callback``) instead.

        """
        future = self._start_read()
        if self.closed():
            self._finish_read(self._read_buffer_size)
            return future
        self._read_until_close = True
        try:
            self._try_inline_read()
        except:
            future.add_done_callback(lambda f: f.exception())
            raise
        return future

    def write(self, data: Union[bytes, memoryview]) -> 'Future[None]':
        """Asynchronously write the given data to this stream.

        This method returns a `.Future` that resolves (with a result
        of ``None``) when the write has been completed.

        The ``data`` argument may be of type `bytes` or `memoryview`.

        .. versionchanged:: 4.0
            Now returns a `.Future` if no callback is given.

        .. versionchanged:: 4.5
            Added support for `memoryview` arguments.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        self._check_closed()
        if data:
            if isinstance(data, memoryview):
                data = memoryview(data).cast('B')
            if self.max_write_buffer_size is not None and len(self._write_buffer) + len(data) > self.max_write_buffer_size:
                raise StreamBufferFullError('Reached maximum write buffer size')
            self._write_buffer.append(data)
            self._total_write_index += len(data)
        future = Future()
        future.add_done_callback(lambda f: f.exception())
        self._write_futures.append((self._total_write_index, future))
        if not self._connecting:
            self._handle_write()
            if self._write_buffer:
                self._add_io_state(self.io_loop.WRITE)
            self._maybe_add_error_listener()
        return future

    def set_close_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Call the given callback when the stream is closed.

        This mostly is not necessary for applications that use the
        `.Future` interface; all outstanding ``Futures`` will resolve
        with a `StreamClosedError` when the stream is closed. However,
        it is still useful as a way to signal that the stream has been
        closed while no other read or write is in progress.

        Unlike other callback-based interfaces, ``set_close_callback``
        was not removed in Tornado 6.0.
        """
        self._close_callback = callback
        self._maybe_add_error_listener()

    def close(self, exc_info: Union[None, bool, BaseException, Tuple['Optional[Type[BaseException]]', Optional[BaseException], Optional[TracebackType]]]=False) -> None:
        """Close this stream.

        If ``exc_info`` is true, set the ``error`` attribute to the current
        exception from `sys.exc_info` (or if ``exc_info`` is a tuple,
        use that instead of `sys.exc_info`).
        """
        if not self.closed():
            if exc_info:
                if isinstance(exc_info, tuple):
                    self.error = exc_info[1]
                elif isinstance(exc_info, BaseException):
                    self.error = exc_info
                else:
                    exc_info = sys.exc_info()
                    if any(exc_info):
                        self.error = exc_info[1]
            if self._read_until_close:
                self._read_until_close = False
                self._finish_read(self._read_buffer_size)
            elif self._read_future is not None:
                try:
                    pos = self._find_read_pos()
                except UnsatisfiableReadError:
                    pass
                else:
                    if pos is not None:
                        self._read_from_buffer(pos)
            if self._state is not None:
                self.io_loop.remove_handler(self.fileno())
                self._state = None
            self.close_fd()
            self._closed = True
        self._signal_closed()

    def _signal_closed(self) -> None:
        futures = []
        if self._read_future is not None:
            futures.append(self._read_future)
            self._read_future = None
        futures += [future for _, future in self._write_futures]
        self._write_futures.clear()
        if self._connect_future is not None:
            futures.append(self._connect_future)
            self._connect_future = None
        for future in futures:
            if not future.done():
                future.set_exception(StreamClosedError(real_error=self.error))
            try:
                future.exception()
            except asyncio.CancelledError:
                pass
        if self._ssl_connect_future is not None:
            if not self._ssl_connect_future.done():
                if self.error is not None:
                    self._ssl_connect_future.set_exception(self.error)
                else:
                    self._ssl_connect_future.set_exception(StreamClosedError())
            self._ssl_connect_future.exception()
            self._ssl_connect_future = None
        if self._close_callback is not None:
            cb = self._close_callback
            self._close_callback = None
            self.io_loop.add_callback(cb)
        self._write_buffer = None

    def reading(self) -> bool:
        """Returns ``True`` if we are currently reading from the stream."""
        return self._read_future is not None

    def writing(self) -> bool:
        """Returns ``True`` if we are currently writing to the stream."""
        return bool(self._write_buffer)

    def closed(self) -> bool:
        """Returns ``True`` if the stream has been closed."""
        return self._closed

    def set_nodelay(self, value: bool) -> None:
        """Sets the no-delay flag for this stream.

        By default, data written to TCP streams may be held for a time
        to make the most efficient use of bandwidth (according to
        Nagle's algorithm).  The no-delay flag requests that data be
        written as soon as possible, even if doing so would consume
        additional bandwidth.

        This flag is currently defined only for TCP-based ``IOStreams``.

        .. versionadded:: 3.1
        """
        pass

    def _handle_connect(self) -> None:
        raise NotImplementedError()

    def _handle_events(self, fd: Union[int, ioloop._Selectable], events: int) -> None:
        if self.closed():
            gen_log.warning('Got events for closed stream %s', fd)
            return
        try:
            if self._connecting:
                self._handle_connect()
            if self.closed():
                return
            if events & self.io_loop.READ:
                self._handle_read()
            if self.closed():
                return
            if events & self.io_loop.WRITE:
                self._handle_write()
            if self.closed():
                return
            if events & self.io_loop.ERROR:
                self.error = self.get_fd_error()
                self.io_loop.add_callback(self.close)
                return
            state = self.io_loop.ERROR
            if self.reading():
                state |= self.io_loop.READ
            if self.writing():
                state |= self.io_loop.WRITE
            if state == self.io_loop.ERROR and self._read_buffer_size == 0:
                state |= self.io_loop.READ
            if state != self._state:
                assert self._state is not None, "shouldn't happen: _handle_events without self._state"
                self._state = state
                self.io_loop.update_handler(self.fileno(), self._state)
        except UnsatisfiableReadError as e:
            gen_log.info('Unsatisfiable read, closing connection: %s' % e)
            self.close(exc_info=e)
        except Exception as e:
            gen_log.error('Uncaught exception, closing connection.', exc_info=True)
            self.close(exc_info=e)
            raise

    def _read_to_buffer_loop(self) -> Optional[int]:
        if self._read_bytes is not None:
            target_bytes = self._read_bytes
        elif self._read_max_bytes is not None:
            target_bytes = self._read_max_bytes
        elif self.reading():
            target_bytes = None
        else:
            target_bytes = 0
        next_find_pos = 0
        while not self.closed():
            if self._read_to_buffer() == 0:
                break
            if target_bytes is not None and self._read_buffer_size >= target_bytes:
                break
            if self._read_buffer_size >= next_find_pos:
                pos = self._find_read_pos()
                if pos is not None:
                    return pos
                next_find_pos = self._read_buffer_size * 2
        return self._find_read_pos()

    def _handle_read(self) -> None:
        try:
            pos = self._read_to_buffer_loop()
        except UnsatisfiableReadError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            gen_log.warning('error on read: %s' % e)
            self.close(exc_info=e)
            return
        if pos is not None:
            self._read_from_buffer(pos)

    def _start_read(self) -> Future:
        if self._read_future is not None:
            self._check_closed()
            assert self._read_future is None, 'Already reading'
        self._read_future = Future()
        return self._read_future

    def _finish_read(self, size: int) -> None:
        if self._user_read_buffer:
            self._read_buffer = self._after_user_read_buffer or bytearray()
            self._after_user_read_buffer = None
            self._read_buffer_size = len(self._read_buffer)
            self._user_read_buffer = False
            result = size
        else:
            result = self._consume(size)
        if self._read_future is not None:
            future = self._read_future
            self._read_future = None
            future_set_result_unless_cancelled(future, result)
        self._maybe_add_error_listener()

    def _try_inline_read(self) -> None:
        """Attempt to complete the current read operation from buffered data.

        If the read can be completed without blocking, schedules the
        read callback on the next IOLoop iteration; otherwise starts
        listening for reads on the socket.
        """
        pos = self._find_read_pos()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        self._check_closed()
        pos = self._read_to_buffer_loop()
        if pos is not None:
            self._read_from_buffer(pos)
            return
        if not self.closed():
            self._add_io_state(ioloop.IOLoop.READ)

    def _read_to_buffer(self) -> Optional[int]:
        """Reads from the socket and appends the result to the read buffer.

        Returns the number of bytes read.  Returns 0 if there is nothing
        to read (i.e. the read returns EWOULDBLOCK or equivalent).  On
        error closes the socket and raises an exception.
        """
        try:
            while True:
                try:
                    if self._user_read_buffer:
                        buf = memoryview(self._read_buffer)[self._read_buffer_size:]
                    else:
                        buf = bytearray(self.read_chunk_size)
                    bytes_read = self.read_from_fd(buf)
                except (socket.error, IOError, OSError) as e:
                    if self._is_connreset(e):
                        self.close(exc_info=e)
                        return None
                    self.close(exc_info=e)
                    raise
                break
            if bytes_read is None:
                return 0
            elif bytes_read == 0:
                self.close()
                return 0
            if not self._user_read_buffer:
                self._read_buffer += memoryview(buf)[:bytes_read]
            self._read_buffer_size += bytes_read
        finally:
            del buf
        if self._read_buffer_size > self.max_buffer_size:
            gen_log.error('Reached maximum read buffer size')
            self.close()
            raise StreamBufferFullError('Reached maximum read buffer size')
        return bytes_read

    def _read_from_buffer(self, pos: int) -> None:
        """Attempts to complete the currently-pending read from the buffer.

        The argument is either a position in the read buffer or None,
        as returned by _find_read_pos.
        """
        self._read_bytes = self._read_delimiter = self._read_regex = None
        self._read_partial = False
        self._finish_read(pos)

    def _find_read_pos(self) -> Optional[int]:
        """Attempts to find a position in the read buffer that satisfies
        the currently-pending read.

        Returns a position in the buffer if the current read can be satisfied,
        or None if it cannot.
        """
        if self._read_bytes is not None and (self._read_buffer_size >= self._read_bytes or (self._read_partial and self._read_buffer_size > 0)):
            num_bytes = min(self._read_bytes, self._read_buffer_size)
            return num_bytes
        elif self._read_delimiter is not None:
            if self._read_buffer:
                loc = self._read_buffer.find(self._read_delimiter)
                if loc != -1:
                    delimiter_len = len(self._read_delimiter)
                    self._check_max_bytes(self._read_delimiter, loc + delimiter_len)
                    return loc + delimiter_len
                self._check_max_bytes(self._read_delimiter, self._read_buffer_size)
        elif self._read_regex is not None:
            if self._read_buffer:
                m = self._read_regex.search(self._read_buffer)
                if m is not None:
                    loc = m.end()
                    self._check_max_bytes(self._read_regex, loc)
                    return loc
                self._check_max_bytes(self._read_regex, self._read_buffer_size)
        return None

    def _check_max_bytes(self, delimiter: Union[bytes, Pattern], size: int) -> None:
        if self._read_max_bytes is not None and size > self._read_max_bytes:
            raise UnsatisfiableReadError('delimiter %r not found within %d bytes' % (delimiter, self._read_max_bytes))

    def _handle_write(self) -> None:
        while True:
            size = len(self._write_buffer)
            if not size:
                break
            assert size > 0
            try:
                if _WINDOWS:
                    size = 128 * 1024
                num_bytes = self.write_to_fd(self._write_buffer.peek(size))
                if num_bytes == 0:
                    break
                self._write_buffer.advance(num_bytes)
                self._total_write_done_index += num_bytes
            except BlockingIOError:
                break
            except (socket.error, IOError, OSError) as e:
                if not self._is_connreset(e):
                    gen_log.warning('Write error on %s: %s', self.fileno(), e)
                self.close(exc_info=e)
                return
        while self._write_futures:
            index, future = self._write_futures[0]
            if index > self._total_write_done_index:
                break
            self._write_futures.popleft()
            future_set_result_unless_cancelled(future, None)

    def _consume(self, loc: int) -> bytes:
        if loc == 0:
            return b''
        assert loc <= self._read_buffer_size
        b = memoryview(self._read_buffer)[:loc].tobytes()
        self._read_buffer_size -= loc
        del self._read_buffer[:loc]
        return b

    def _check_closed(self) -> None:
        if self.closed():
            raise StreamClosedError(real_error=self.error)

    def _maybe_add_error_listener(self) -> None:
        if self._state is None or self._state == ioloop.IOLoop.ERROR:
            if not self.closed() and self._read_buffer_size == 0 and (self._close_callback is not None):
                self._add_io_state(ioloop.IOLoop.READ)

    def _add_io_state(self, state: int) -> None:
        """Adds `state` (IOLoop.{READ,WRITE} flags) to our event handler.

        Implementation notes: Reads and writes have a fast path and a
        slow path.  The fast path reads synchronously from socket
        buffers, while the slow path uses `_add_io_state` to schedule
        an IOLoop callback.

        To detect closed connections, we must have called
        `_add_io_state` at some point, but we want to delay this as
        much as possible so we don't have to set an `IOLoop.ERROR`
        listener that will be overwritten by the next slow-path
        operation. If a sequence of fast-path ops do not end in a
        slow-path op, (e.g. for an @asynchronous long-poll request),
        we must add the error handler.

        TODO: reevaluate this now that callbacks are gone.

        """
        if self.closed():
            return
        if self._state is None:
            self._state = ioloop.IOLoop.ERROR | state
            self.io_loop.add_handler(self.fileno(), self._handle_events, self._state)
        elif not self._state & state:
            self._state = self._state | state
            self.io_loop.update_handler(self.fileno(), self._state)

    def _is_connreset(self, exc: BaseException) -> bool:
        """Return ``True`` if exc is ECONNRESET or equivalent.

        May be overridden in subclasses.
        """
        return isinstance(exc, (socket.error, IOError)) and errno_from_exception(exc) in _ERRNO_CONNRESET
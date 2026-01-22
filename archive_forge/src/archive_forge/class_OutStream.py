import asyncio
import atexit
import contextvars
import io
import os
import sys
import threading
import traceback
import warnings
from binascii import b2a_hex
from collections import defaultdict, deque
from io import StringIO, TextIOBase
from threading import local
from typing import Any, Callable, Deque, Dict, Optional
import zmq
from jupyter_client.session import extract_header
from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
class OutStream(TextIOBase):
    """A file like object that publishes the stream to a 0MQ PUB socket.

    Output is handed off to an IO Thread
    """
    flush_timeout = 10
    flush_interval = 0.2
    topic = None
    encoding = 'UTF-8'
    _exc: Optional[Any] = None

    def fileno(self):
        """
        Things like subprocess will peak and write to the fileno() of stderr/stdout.
        """
        if getattr(self, '_original_stdstream_copy', None) is not None:
            return self._original_stdstream_copy
        msg = 'fileno'
        raise io.UnsupportedOperation(msg)

    def _watch_pipe_fd(self):
        """
        We've redirected standards streams 0 and 1 into a pipe.

        We need to watch in a thread and redirect them to the right places.

        1) the ZMQ channels to show in notebook interfaces,
        2) the original stdout/err, to capture errors in terminals.

        We cannot schedule this on the ioloop thread, as this might be blocking.

        """
        try:
            bts = os.read(self._fid, PIPE_BUFFER_SIZE)
            while bts and self._should_watch:
                self.write(bts.decode(errors='replace'))
                os.write(self._original_stdstream_copy, bts)
                bts = os.read(self._fid, PIPE_BUFFER_SIZE)
        except Exception:
            self._exc = sys.exc_info()

    def __init__(self, session, pub_thread, name, pipe=None, echo=None, *, watchfd=True, isatty=False):
        """
        Parameters
        ----------
        session : object
            the session object
        pub_thread : threading.Thread
            the publication thread
        name : str {'stderr', 'stdout'}
            the name of the standard stream to replace
        pipe : object
            the pipe object
        echo : bool
            whether to echo output
        watchfd : bool (default, True)
            Watch the file descriptor corresponding to the replaced stream.
            This is useful if you know some underlying code will write directly
            the file descriptor by its number. It will spawn a watching thread,
            that will swap the give file descriptor for a pipe, read from the
            pipe, and insert this into the current Stream.
        isatty : bool (default, False)
            Indication of whether this stream has terminal capabilities (e.g. can handle colors)

        """
        if pipe is not None:
            warnings.warn('pipe argument to OutStream is deprecated and ignored since ipykernel 4.2.3.', DeprecationWarning, stacklevel=2)
        self.session = session
        if not isinstance(pub_thread, IOPubThread):
            warnings.warn('Since IPykernel 4.3, OutStream should be created with IOPubThread, not %r' % pub_thread, DeprecationWarning, stacklevel=2)
            pub_thread = IOPubThread(pub_thread)
            pub_thread.start()
        self.pub_thread = pub_thread
        self.name = name
        self.topic = b'stream.' + name.encode()
        self._parent_header: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('parent_header')
        self._parent_header.set({})
        self._thread_to_parent = {}
        self._thread_to_parent_header = {}
        self._parent_header_global = {}
        self._master_pid = os.getpid()
        self._flush_pending = False
        self._subprocess_flush_pending = False
        self._io_loop = pub_thread.io_loop
        self._buffer_lock = threading.RLock()
        self._buffers = defaultdict(StringIO)
        self.echo = None
        self._isatty = bool(isatty)
        self._should_watch = False
        self._local = local()
        if watchfd and ((sys.platform.startswith('linux') or sys.platform.startswith('darwin')) and 'PYTEST_CURRENT_TEST' not in os.environ) or watchfd == 'force':
            self._should_watch = True
            self._setup_stream_redirects(name)
        if echo:
            if hasattr(echo, 'read') and hasattr(echo, 'write'):
                if self._should_watch:
                    try:
                        echo_fd = echo.fileno()
                    except Exception:
                        echo_fd = None
                    if echo_fd is not None and echo_fd == self._original_stdstream_fd:
                        echo = io.TextIOWrapper(io.FileIO(self._original_stdstream_copy, 'w'))
                self.echo = echo
            else:
                msg = 'echo argument must be a file-like object'
                raise ValueError(msg)

    @property
    def parent_header(self):
        try:
            return self._parent_header.get()
        except LookupError:
            try:
                identity = threading.current_thread().ident
                while identity in self._thread_to_parent:
                    identity = self._thread_to_parent[identity]
                return self._thread_to_parent_header[identity]
            except KeyError:
                return self._parent_header_global

    @parent_header.setter
    def parent_header(self, value):
        self._parent_header_global = value
        return self._parent_header.set(value)

    def isatty(self):
        """Return a bool indicating whether this is an 'interactive' stream.

        Returns:
            Boolean
        """
        return self._isatty

    def _setup_stream_redirects(self, name):
        pr, pw = os.pipe()
        fno = self._original_stdstream_fd = getattr(sys, name).fileno()
        self._original_stdstream_copy = os.dup(fno)
        os.dup2(pw, fno)
        self._fid = pr
        self._exc = None
        self.watch_fd_thread = threading.Thread(target=self._watch_pipe_fd)
        self.watch_fd_thread.daemon = True
        self.watch_fd_thread.start()

    def _is_master_process(self):
        return os.getpid() == self._master_pid

    def set_parent(self, parent):
        """Set the parent header."""
        self.parent_header = extract_header(parent)

    def close(self):
        """Close the stream."""
        if self._should_watch:
            self._should_watch = False
            os.write(self._original_stdstream_fd, b'\x00')
            self.watch_fd_thread.join()
            os.dup2(self._original_stdstream_copy, self._original_stdstream_fd)
            os.close(self._original_stdstream_copy)
        if self._exc:
            etype, value, tb = self._exc
            traceback.print_exception(etype, value, tb)
        self.pub_thread = None

    @property
    def closed(self):
        return self.pub_thread is None

    def _schedule_flush(self):
        """schedule a flush in the IO thread

        call this on write, to indicate that flush should be called soon.
        """
        if self._flush_pending:
            return
        self._flush_pending = True

        def _schedule_in_thread():
            self._io_loop.call_later(self.flush_interval, self._flush)
        self.pub_thread.schedule(_schedule_in_thread)

    def flush(self):
        """trigger actual zmq send

        send will happen in the background thread
        """
        if self.pub_thread and self.pub_thread.thread is not None and self.pub_thread.thread.is_alive() and (self.pub_thread.thread.ident != threading.current_thread().ident):
            self.pub_thread.schedule(self._flush)
            evt = threading.Event()
            self.pub_thread.schedule(evt.set)
            if not evt.wait(self.flush_timeout):
                print('IOStream.flush timed out', file=sys.__stderr__)
        else:
            self._flush()

    def _flush(self):
        """This is where the actual send happens.

        _flush should generally be called in the IO thread,
        unless the thread has been destroyed (e.g. forked subprocess).
        """
        self._flush_pending = False
        self._subprocess_flush_pending = False
        if self.echo is not None:
            try:
                self.echo.flush()
            except OSError as e:
                if self.echo is not sys.__stderr__:
                    print(f'Flush failed: {e}', file=sys.__stderr__)
        for parent, data in self._flush_buffers():
            if data:
                self.session.pid = os.getpid()
                content = {'name': self.name, 'text': data}
                msg = self.session.msg('stream', content, parent=parent)
                for hook in self._hooks:
                    msg = hook(msg)
                    if msg is None:
                        return
                self.session.send(self.pub_thread, msg, ident=self.topic)

    def write(self, string: str) -> Optional[int]:
        """Write to current stream after encoding if necessary

        Returns
        -------
        len : int
            number of items from input parameter written to stream.

        """
        parent = self.parent_header
        if not isinstance(string, str):
            msg = f'write() argument must be str, not {type(string)}'
            raise TypeError(msg)
        if self.echo is not None:
            try:
                self.echo.write(string)
            except OSError as e:
                if self.echo is not sys.__stderr__:
                    print(f'Write failed: {e}', file=sys.__stderr__)
        if self.pub_thread is None:
            msg = 'I/O operation on closed file'
            raise ValueError(msg)
        is_child = not self._is_master_process()
        with self._buffer_lock:
            self._buffers[frozenset(parent.items())].write(string)
        if is_child:
            if self._subprocess_flush_pending:
                return None
            self._subprocess_flush_pending = True
            self.pub_thread.schedule(self._flush)
        else:
            self._schedule_flush()
        return len(string)

    def writelines(self, sequence):
        """Write lines to the stream."""
        if self.pub_thread is None:
            msg = 'I/O operation on closed file'
            raise ValueError(msg)
        for string in sequence:
            self.write(string)

    def writable(self):
        """Test whether the stream is writable."""
        return True

    def _flush_buffers(self):
        """clear the current buffer and return the current buffer data."""
        buffers = self._rotate_buffers()
        for frozen_parent, buffer in buffers.items():
            data = buffer.getvalue()
            buffer.close()
            yield (dict(frozen_parent), data)

    def _rotate_buffers(self):
        """Returns the current buffer and replaces it with an empty buffer."""
        with self._buffer_lock:
            old_buffers = self._buffers
            self._buffers = defaultdict(StringIO)
        return old_buffers

    @property
    def _hooks(self):
        if not hasattr(self._local, 'hooks'):
            self._local.hooks = []
        return self._local.hooks

    def register_hook(self, hook):
        """
        Registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object

        Returns
        -------
        Either a publishable message, or `None`.
        The hook callable must return a message from
        the __call__ method if they still require the
        `session.send` method to be called after transformation.
        Returning `None` will halt that execution path, and
        session.send will not be called.
        """
        self._hooks.append(hook)

    def unregister_hook(self, hook):
        """
        Un-registers a hook with the thread-local storage.

        Parameters
        ----------
        hook : Any callable object which has previously been
            registered as a hook.

        Returns
        -------
        bool - `True` if the hook was removed, `False` if it wasn't
            found.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False
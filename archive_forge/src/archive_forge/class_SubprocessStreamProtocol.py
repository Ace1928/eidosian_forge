import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
class SubprocessStreamProtocol(streams.FlowControlMixin, protocols.SubprocessProtocol):
    """Like StreamReaderProtocol, but for a subprocess."""

    def __init__(self, limit, loop):
        super().__init__(loop=loop)
        self._limit = limit
        self.stdin = self.stdout = self.stderr = None
        self._transport = None
        self._process_exited = False
        self._pipe_fds = []
        self._stdin_closed = self._loop.create_future()

    def __repr__(self):
        info = [self.__class__.__name__]
        if self.stdin is not None:
            info.append(f'stdin={self.stdin!r}')
        if self.stdout is not None:
            info.append(f'stdout={self.stdout!r}')
        if self.stderr is not None:
            info.append(f'stderr={self.stderr!r}')
        return '<{}>'.format(' '.join(info))

    def connection_made(self, transport):
        self._transport = transport
        stdout_transport = transport.get_pipe_transport(1)
        if stdout_transport is not None:
            self.stdout = streams.StreamReader(limit=self._limit, loop=self._loop)
            self.stdout.set_transport(stdout_transport)
            self._pipe_fds.append(1)
        stderr_transport = transport.get_pipe_transport(2)
        if stderr_transport is not None:
            self.stderr = streams.StreamReader(limit=self._limit, loop=self._loop)
            self.stderr.set_transport(stderr_transport)
            self._pipe_fds.append(2)
        stdin_transport = transport.get_pipe_transport(0)
        if stdin_transport is not None:
            self.stdin = streams.StreamWriter(stdin_transport, protocol=self, reader=None, loop=self._loop)

    def pipe_data_received(self, fd, data):
        if fd == 1:
            reader = self.stdout
        elif fd == 2:
            reader = self.stderr
        else:
            reader = None
        if reader is not None:
            reader.feed_data(data)

    def pipe_connection_lost(self, fd, exc):
        if fd == 0:
            pipe = self.stdin
            if pipe is not None:
                pipe.close()
            self.connection_lost(exc)
            if exc is None:
                self._stdin_closed.set_result(None)
            else:
                self._stdin_closed.set_exception(exc)
                self._stdin_closed._log_traceback = False
            return
        if fd == 1:
            reader = self.stdout
        elif fd == 2:
            reader = self.stderr
        else:
            reader = None
        if reader is not None:
            if exc is None:
                reader.feed_eof()
            else:
                reader.set_exception(exc)
        if fd in self._pipe_fds:
            self._pipe_fds.remove(fd)
        self._maybe_close_transport()

    def process_exited(self):
        self._process_exited = True
        self._maybe_close_transport()

    def _maybe_close_transport(self):
        if len(self._pipe_fds) == 0 and self._process_exited:
            self._transport.close()
            self._transport = None

    def _get_close_waiter(self, stream):
        if stream is self.stdin:
            return self._stdin_closed
from __future__ import unicode_literals
from .base import EventLoop, INPUT_TIMEOUT
from ..terminal.win32_input import ConsoleInputReader
from .callbacks import EventLoopCallbacks
from .asyncio_base import AsyncioTimeout
import asyncio
class Win32AsyncioEventLoop(EventLoop):

    def __init__(self, loop=None):
        self._console_input_reader = ConsoleInputReader()
        self.running = False
        self.closed = False
        self.loop = loop or asyncio.get_event_loop()

    @asyncio.coroutine
    def run_as_coroutine(self, stdin, callbacks):
        """
        The input 'event loop'.
        """
        assert isinstance(callbacks, EventLoopCallbacks)
        if self.closed:
            raise Exception('Event loop already closed.')
        timeout = AsyncioTimeout(INPUT_TIMEOUT, callbacks.input_timeout, self.loop)
        self.running = True
        try:
            while self.running:
                timeout.reset()
                try:
                    g = iter(self.loop.run_in_executor(None, self._console_input_reader.read))
                    while True:
                        yield next(g)
                except StopIteration as e:
                    keys = e.args[0]
                for k in keys:
                    callbacks.feed_key(k)
        finally:
            timeout.stop()

    def stop(self):
        self.running = False

    def close(self):
        self.closed = True
        self._console_input_reader.close()

    def run_in_executor(self, callback):
        self.loop.run_in_executor(None, callback)

    def call_from_executor(self, callback, _max_postpone_until=None):
        self.loop.call_soon_threadsafe(callback)

    def add_reader(self, fd, callback):
        """ Start watching the file descriptor for read availability. """
        self.loop.add_reader(fd, callback)

    def remove_reader(self, fd):
        """ Stop watching the file descriptor for read availability. """
        self.loop.remove_reader(fd)
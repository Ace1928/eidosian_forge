from __future__ import unicode_literals
import fcntl
import os
import random
import signal
import threading
import time
from prompt_toolkit.terminal.vt100_input import InputStream
from prompt_toolkit.utils import DummyContext, in_main_thread
from prompt_toolkit.input import Input
from .base import EventLoop, INPUT_TIMEOUT
from .callbacks import EventLoopCallbacks
from .inputhook import InputHookContext
from .posix_utils import PosixStdinReader
from .utils import TimeIt
from .select import AutoSelector, Selector, fd_to_int
class PosixEventLoop(EventLoop):
    """
    Event loop for posix systems (Linux, Mac os X).
    """

    def __init__(self, inputhook=None, selector=AutoSelector):
        assert inputhook is None or callable(inputhook)
        assert issubclass(selector, Selector)
        self.running = False
        self.closed = False
        self._running = False
        self._callbacks = None
        self._calls_from_executor = []
        self._read_fds = {}
        self.selector = selector()
        self._schedule_pipe = os.pipe()
        fcntl.fcntl(self._schedule_pipe[0], fcntl.F_SETFL, os.O_NONBLOCK)
        self._inputhook_context = InputHookContext(inputhook) if inputhook else None

    def run(self, stdin, callbacks):
        """
        The input 'event loop'.
        """
        assert isinstance(stdin, Input)
        assert isinstance(callbacks, EventLoopCallbacks)
        assert not self._running
        if self.closed:
            raise Exception('Event loop already closed.')
        self._running = True
        self._callbacks = callbacks
        inputstream = InputStream(callbacks.feed_key)
        current_timeout = [INPUT_TIMEOUT]
        stdin_reader = PosixStdinReader(stdin.fileno())
        if in_main_thread():
            ctx = call_on_sigwinch(self.received_winch)
        else:
            ctx = DummyContext()

        def read_from_stdin():
            """ Read user input. """
            data = stdin_reader.read()
            inputstream.feed(data)
            current_timeout[0] = INPUT_TIMEOUT
            if stdin_reader.closed:
                self.stop()
        self.add_reader(stdin, read_from_stdin)
        self.add_reader(self._schedule_pipe[0], None)
        with ctx:
            while self._running:
                if self._inputhook_context:
                    with TimeIt() as inputhook_timer:

                        def ready(wait):
                            """ True when there is input ready. The inputhook should return control. """
                            return self._ready_for_reading(current_timeout[0] if wait else 0) != []
                        self._inputhook_context.call_inputhook(ready)
                    inputhook_duration = inputhook_timer.duration
                else:
                    inputhook_duration = 0
                if current_timeout[0] is None:
                    remaining_timeout = None
                else:
                    remaining_timeout = max(0, current_timeout[0] - inputhook_duration)
                fds = self._ready_for_reading(remaining_timeout)
                if fds:
                    tasks = []
                    low_priority_tasks = []
                    now = None
                    for fd in fds:
                        if fd == self._schedule_pipe[0]:
                            for c, max_postpone_until in self._calls_from_executor:
                                if max_postpone_until is None:
                                    tasks.append(c)
                                else:
                                    now = now or _now()
                                    if max_postpone_until < now:
                                        tasks.append(c)
                                    else:
                                        low_priority_tasks.append((c, max_postpone_until))
                            self._calls_from_executor = []
                            os.read(self._schedule_pipe[0], 1024)
                        else:
                            handler = self._read_fds.get(fd)
                            if handler:
                                tasks.append(handler)
                    random.shuffle(tasks)
                    random.shuffle(low_priority_tasks)
                    if tasks:
                        for t in tasks:
                            t()
                        for t, max_postpone_until in low_priority_tasks:
                            self.call_from_executor(t, _max_postpone_until=max_postpone_until)
                    else:
                        for t, _ in low_priority_tasks:
                            t()
                else:
                    inputstream.flush()
                    callbacks.input_timeout()
                    current_timeout[0] = None
        self.remove_reader(stdin)
        self.remove_reader(self._schedule_pipe[0])
        self._callbacks = None

    def _ready_for_reading(self, timeout=None):
        """
        Return the file descriptors that are ready for reading.
        """
        fds = self.selector.select(timeout)
        return fds

    def received_winch(self):
        """
        Notify the event loop that SIGWINCH has been received
        """

        def process_winch():
            if self._callbacks:
                self._callbacks.terminal_size_changed()
        self.call_from_executor(process_winch)

    def run_in_executor(self, callback):
        """
        Run a long running function in a background thread.
        (This is recommended for code that could block the event loop.)
        Similar to Twisted's ``deferToThread``.
        """

        def start_executor():
            threading.Thread(target=callback).start()
        self.call_from_executor(start_executor)

    def call_from_executor(self, callback, _max_postpone_until=None):
        """
        Call this function in the main event loop.
        Similar to Twisted's ``callFromThread``.

        :param _max_postpone_until: `None` or `time.time` value. For interal
            use. If the eventloop is saturated, consider this task to be low
            priority and postpone maximum until this timestamp. (For instance,
            repaint is done using low priority.)
        """
        assert _max_postpone_until is None or isinstance(_max_postpone_until, float)
        self._calls_from_executor.append((callback, _max_postpone_until))
        if self._schedule_pipe:
            try:
                os.write(self._schedule_pipe[1], b'x')
            except (AttributeError, IndexError, OSError):
                pass

    def stop(self):
        """
        Stop the event loop.
        """
        self._running = False

    def close(self):
        self.closed = True
        schedule_pipe = self._schedule_pipe
        self._schedule_pipe = None
        if schedule_pipe:
            os.close(schedule_pipe[0])
            os.close(schedule_pipe[1])
        if self._inputhook_context:
            self._inputhook_context.close()

    def add_reader(self, fd, callback):
        """ Add read file descriptor to the event loop. """
        fd = fd_to_int(fd)
        self._read_fds[fd] = callback
        self.selector.register(fd)

    def remove_reader(self, fd):
        """ Remove read file descriptor from the event loop. """
        fd = fd_to_int(fd)
        if fd in self._read_fds:
            del self._read_fds[fd]
        self.selector.unregister(fd)
import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
class _UnixDefaultEventLoopPolicy(events.BaseDefaultEventLoopPolicy):
    """UNIX event loop policy with a watcher for child processes."""
    _loop_factory = _UnixSelectorEventLoop

    def __init__(self):
        super().__init__()
        self._watcher = None

    def _init_watcher(self):
        with events._lock:
            if self._watcher is None:
                self._watcher = ThreadedChildWatcher()

    def set_event_loop(self, loop):
        """Set the event loop.

        As a side effect, if a child watcher was set before, then calling
        .set_event_loop() from the main thread will call .attach_loop(loop) on
        the child watcher.
        """
        super().set_event_loop(loop)
        if self._watcher is not None and threading.current_thread() is threading.main_thread():
            self._watcher.attach_loop(loop)

    def get_child_watcher(self):
        """Get the watcher for child processes.

        If not yet set, a ThreadedChildWatcher object is automatically created.
        """
        if self._watcher is None:
            self._init_watcher()
        return self._watcher

    def set_child_watcher(self, watcher):
        """Set the watcher for child processes."""
        assert watcher is None or isinstance(watcher, AbstractChildWatcher)
        if self._watcher is not None:
            self._watcher.close()
        self._watcher = watcher
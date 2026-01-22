import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
class PoolThread(DummyProcess):

    def __init__(self, *args, **kwargs):
        DummyProcess.__init__(self)
        self._state = RUN
        self._was_started = False
        self.daemon = True

    def run(self):
        try:
            return self.body()
        except RestartFreqExceeded as exc:
            error('Thread %r crashed: %r', type(self).__name__, exc, exc_info=1)
            _kill(os.getpid(), TERM_SIGNAL)
            sys.exit()
        except Exception as exc:
            error('Thread %r crashed: %r', type(self).__name__, exc, exc_info=1)
            os._exit(1)

    def start(self, *args, **kwargs):
        self._was_started = True
        super(PoolThread, self).start(*args, **kwargs)

    def on_stop_not_started(self):
        pass

    def stop(self, timeout=None):
        if self._was_started:
            self.join(timeout)
            return
        self.on_stop_not_started()

    def terminate(self):
        self._state = TERMINATE

    def close(self):
        self._state = CLOSE
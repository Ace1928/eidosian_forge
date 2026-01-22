import time
import warnings
from typing import Tuple
from zmq import ETERM, POLLERR, POLLIN, POLLOUT, Poller, ZMQError
from .minitornado.ioloop import PeriodicCallback, PollIOLoop
from .minitornado.log import gen_log
class DelayedCallback(PeriodicCallback):
    """Schedules the given callback to be called once.

    The callback is called once, after callback_time milliseconds.

    `start` must be called after the DelayedCallback is created.

    The timeout is calculated from when `start` is called.
    """

    def __init__(self, callback, callback_time, io_loop=None):
        warnings.warn('DelayedCallback is deprecated.\n        Use loop.add_timeout instead.', DeprecationWarning)
        callback_time = max(callback_time, 0.001)
        super().__init__(callback, callback_time, io_loop)

    def start(self):
        """Starts the timer."""
        self._running = True
        self._firstrun = True
        self._next_timeout = time.time() + self.callback_time / 1000.0
        self.io_loop.add_timeout(self._next_timeout, self._run)

    def _run(self):
        if not self._running:
            return
        self._running = False
        try:
            self.callback()
        except Exception:
            gen_log.error('Error in delayed callback', exc_info=True)
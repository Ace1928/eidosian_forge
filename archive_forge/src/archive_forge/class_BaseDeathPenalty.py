import ctypes
import signal
import threading
class BaseDeathPenalty:
    """Base class to setup job timeouts."""

    def __init__(self, timeout, exception=BaseTimeoutException, **kwargs):
        self._timeout = timeout
        self._exception = exception

    def __enter__(self):
        self.setup_death_penalty()

    def __exit__(self, type, value, traceback):
        try:
            self.cancel_death_penalty()
        except BaseTimeoutException:
            pass
        return False

    def setup_death_penalty(self):
        raise NotImplementedError()

    def cancel_death_penalty(self):
        raise NotImplementedError()
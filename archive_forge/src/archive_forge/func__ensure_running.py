import os
import time
from threading import Thread, Lock
import sentry_sdk
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
def _ensure_running(self):
    """
        Check that the monitor has an active thread to run in, or create one if not.

        Note that this might fail (e.g. in Python 3.12 it's not possible to
        spawn new threads at interpreter shutdown). In that case self._running
        will be False after running this function.
        """
    if self._thread_for_pid == os.getpid() and self._thread is not None:
        return None
    with self._thread_lock:
        if self._thread_for_pid == os.getpid() and self._thread is not None:
            return None

        def _thread():
            while self._running:
                time.sleep(self.interval)
                if self._running:
                    self.run()
        thread = Thread(name=self.name, target=_thread)
        thread.daemon = True
        try:
            thread.start()
        except RuntimeError:
            self._running = False
            return None
        self._thread = thread
        self._thread_for_pid = os.getpid()
    return None
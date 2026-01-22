import eventlet.hubs
from eventlet.patcher import slurp_properties
from eventlet.support import greenlets as greenlet
from collections import deque
class _BlockedThread:
    """Is either empty, or represents a single blocked thread that
    blocked itself by calling the block() method. The thread can be
    awoken by calling wake(). Wake() can be called multiple times and
    all but the first call will have no effect."""

    def __init__(self):
        self._blocked_thread = None
        self._wakeupper = None
        self._hub = eventlet.hubs.get_hub()

    def __nonzero__(self):
        return self._blocked_thread is not None
    __bool__ = __nonzero__

    def block(self, deadline=None):
        if self._blocked_thread is not None:
            raise Exception('Cannot block more than one thread on one BlockedThread')
        self._blocked_thread = greenlet.getcurrent()
        if deadline is not None:
            self._hub.schedule_call_local(deadline - self._hub.clock(), self.wake)
        try:
            self._hub.switch()
        finally:
            self._blocked_thread = None
            if self._wakeupper is not None:
                self._wakeupper.cancel()
                self._wakeupper = None

    def wake(self):
        """Schedules the blocked thread to be awoken and return
        True. If wake has already been called or if there is no
        blocked thread, then this call has no effect and returns
        False."""
        if self._blocked_thread is not None and self._wakeupper is None:
            self._wakeupper = self._hub.schedule_call_global(0, self._blocked_thread.switch)
            return True
        return False
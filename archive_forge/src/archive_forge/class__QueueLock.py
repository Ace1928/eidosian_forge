import eventlet.hubs
from eventlet.patcher import slurp_properties
from eventlet.support import greenlets as greenlet
from collections import deque
class _QueueLock:
    """A Lock that can be acquired by at most one thread. Any other
    thread calling acquire will be blocked in a queue. When release
    is called, the threads are awoken in the order they blocked,
    one at a time. This lock can be required recursively by the same
    thread."""

    def __init__(self):
        self._waiters = deque()
        self._count = 0
        self._holder = None
        self._hub = eventlet.hubs.get_hub()

    def __nonzero__(self):
        return bool(self._count)
    __bool__ = __nonzero__

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

    def acquire(self):
        current = greenlet.getcurrent()
        if (self._waiters or self._count > 0) and self._holder is not current:
            self._waiters.append(current)
            self._hub.switch()
            w = self._waiters.popleft()
            assert w is current, 'Waiting threads woken out of order'
            assert self._count == 0, 'After waking a thread, the lock must be unacquired'
        self._holder = current
        self._count += 1

    def release(self):
        if self._count <= 0:
            raise LockReleaseError('Cannot release unacquired lock')
        self._count -= 1
        if self._count == 0:
            self._holder = None
            if self._waiters:
                self._hub.schedule_call_global(0, self._waiters[0].switch)
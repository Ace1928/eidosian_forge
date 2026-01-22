import collections
import eventlet
from eventlet import hubs
def _do_acquire(self):
    if self._waiters and self.counter > 0:
        waiter = self._waiters.popleft()
        waiter.switch()
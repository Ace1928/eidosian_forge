import collections
import heapq
from types import GenericAlias
from . import locks
from . import mixins
def _wakeup_next(self, waiters):
    while waiters:
        waiter = waiters.popleft()
        if not waiter.done():
            waiter.set_result(None)
            break
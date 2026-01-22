import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
class GreenPile:
    """GreenPile is an abstraction representing a bunch of I/O-related tasks.

    Construct a GreenPile with an existing GreenPool object.  The GreenPile will
    then use that pool's concurrency as it processes its jobs.  There can be
    many GreenPiles associated with a single GreenPool.

    A GreenPile can also be constructed standalone, not associated with any
    GreenPool.  To do this, construct it with an integer size parameter instead
    of a GreenPool.

    It is not advisable to iterate over a GreenPile in a different greenthread
    than the one which is calling spawn.  The iterator will exit early in that
    situation.
    """

    def __init__(self, size_or_pool=1000):
        if isinstance(size_or_pool, GreenPool):
            self.pool = size_or_pool
        else:
            self.pool = GreenPool(size_or_pool)
        self.waiters = queue.LightQueue()
        self.counter = 0

    def spawn(self, func, *args, **kw):
        """Runs *func* in its own green thread, with the result available by
        iterating over the GreenPile object."""
        self.counter += 1
        try:
            gt = self.pool.spawn(func, *args, **kw)
            self.waiters.put(gt)
        except:
            self.counter -= 1
            raise

    def __iter__(self):
        return self

    def next(self):
        """Wait for the next result, suspending the current greenthread until it
        is available.  Raises StopIteration when there are no more results."""
        if self.counter == 0:
            raise StopIteration()
        return self._next()
    __next__ = next

    def _next(self):
        try:
            return self.waiters.get().wait()
        finally:
            self.counter -= 1
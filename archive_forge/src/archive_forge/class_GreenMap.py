import traceback
import eventlet
from eventlet import queue
from eventlet.support import greenlets as greenlet
class GreenMap(GreenPile):

    def __init__(self, size_or_pool):
        super().__init__(size_or_pool)
        self.waiters = queue.LightQueue(maxsize=self.pool.size)

    def done_spawning(self):
        self.spawn(lambda: StopIteration())

    def next(self):
        val = self._next()
        if isinstance(val, StopIteration):
            raise val
        else:
            return val
    __next__ = next
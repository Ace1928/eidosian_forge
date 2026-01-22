from collections import deque
from threading import local
def _async_invoke_later(self, fn, scheduler):
    self.late_queue.append(fn)
    self.queue_tick(scheduler)
import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
class DebugListener(FdListener):

    def __init__(self, evtype, fileno, cb, tb, mark_as_closed):
        self.where_called = traceback.format_stack()
        self.greenlet = greenlet.getcurrent()
        super().__init__(evtype, fileno, cb, tb, mark_as_closed)

    def __repr__(self):
        return 'DebugListener(%r, %r, %r, %r, %r, %r)\n%sEndDebugFdListener' % (self.evtype, self.fileno, self.cb, self.tb, self.mark_as_closed, self.greenlet, ''.join(self.where_called))
    __str__ = __repr__
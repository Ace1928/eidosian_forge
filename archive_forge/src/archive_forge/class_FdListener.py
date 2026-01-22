import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
class FdListener:

    def __init__(self, evtype, fileno, cb, tb, mark_as_closed):
        """ The following are required:
        cb - the standard callback, which will switch into the
            listening greenlet to indicate that the event waited upon
            is ready
        tb - a 'throwback'. This is typically greenlet.throw, used
            to raise a signal into the target greenlet indicating that
            an event was obsoleted by its underlying filehandle being
            repurposed.
        mark_as_closed - if any listener is obsoleted, this is called
            (in the context of some other client greenlet) to alert
            underlying filehandle-wrapping objects that they've been
            closed.
        """
        assert evtype is READ or evtype is WRITE
        self.evtype = evtype
        self.fileno = fileno
        self.cb = cb
        self.tb = tb
        self.mark_as_closed = mark_as_closed
        self.spent = False
        self.greenlet = greenlet.getcurrent()

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (type(self).__name__, self.evtype, self.fileno, self.cb, self.tb)
    __str__ = __repr__

    def defang(self):
        self.cb = closed_callback
        if self.mark_as_closed is not None:
            self.mark_as_closed()
        self.spent = True
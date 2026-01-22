import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def add_timer(self, timer):
    scheduled_time = self.clock() + timer.seconds
    self.next_timers.append((scheduled_time, timer))
    return scheduled_time
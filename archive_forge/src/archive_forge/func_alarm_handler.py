import errno
import heapq
import math
import signal
import sys
import traceback
import eventlet.hubs
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
def alarm_handler(signum, frame):
    import inspect
    raise RuntimeError('Blocking detector ALARMED at' + str(inspect.getframeinfo(frame)))
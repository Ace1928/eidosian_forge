from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
Kills the greenthread using :func:`kill`, but only if it hasn't
        already started running.  After being canceled,
        all calls to :meth:`wait` will raise *throw_args* (which default
        to :class:`greenlet.GreenletExit`).
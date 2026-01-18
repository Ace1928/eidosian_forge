import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def wait_for(self, predicate, timeout=None):
    result = predicate()
    if result:
        return result
    if timeout is not None:
        endtime = getattr(time, 'monotonic', time.time)() + timeout
    else:
        endtime = None
        waittime = None
    while not result:
        if endtime is not None:
            waittime = endtime - getattr(time, 'monotonic', time.time)()
            if waittime <= 0:
                break
        self.wait(waittime)
        result = predicate()
    return result
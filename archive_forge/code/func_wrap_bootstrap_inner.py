import _thread as __thread
from eventlet.support import greenlets as greenlet
from eventlet import greenthread
from eventlet.lock import Lock
import sys
from eventlet.corolocal import local as _local
def wrap_bootstrap_inner():
    try:
        bootstrap_inner()
    finally:
        if thread._tstate_lock is not None:
            thread._tstate_lock.release()
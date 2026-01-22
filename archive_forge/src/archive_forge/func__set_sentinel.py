import _thread as __thread
from eventlet.support import greenlets as greenlet
from eventlet import greenthread
from eventlet.lock import Lock
import sys
from eventlet.corolocal import local as _local
def _set_sentinel():
    return allocate_lock()
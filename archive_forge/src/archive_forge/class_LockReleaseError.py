import eventlet.hubs
from eventlet.patcher import slurp_properties
from eventlet.support import greenlets as greenlet
from collections import deque
class LockReleaseError(Exception):
    pass
import eventlet.hubs
from eventlet.patcher import slurp_properties
from eventlet.support import greenlets as greenlet
from collections import deque
A recv_pyobj method that's safe to use when multiple
        greenthreads are calling send, send_pyobj, recv and
        recv_pyobj on the same socket.
        
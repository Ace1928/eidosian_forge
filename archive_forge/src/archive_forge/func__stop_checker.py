import sys
import warnings
from eventlet import greenpool
from eventlet import greenthread
from eventlet import support
from eventlet.green import socket
from eventlet.support import greenlets as greenlet
def _stop_checker(t, server_gt, conn):
    try:
        try:
            t.wait()
        finally:
            conn.close()
    except greenlet.GreenletExit:
        pass
    except Exception:
        greenthread.kill(server_gt, *sys.exc_info())
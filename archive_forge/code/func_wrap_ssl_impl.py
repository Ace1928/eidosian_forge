import sys
import warnings
from eventlet import greenpool
from eventlet import greenthread
from eventlet import support
from eventlet.green import socket
from eventlet.support import greenlets as greenlet
def wrap_ssl_impl(*a, **kw):
    raise ImportError('To use SSL with Eventlet, you must install PyOpenSSL or use Python 2.7 or later.')
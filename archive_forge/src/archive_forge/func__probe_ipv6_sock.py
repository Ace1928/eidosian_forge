from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
def _probe_ipv6_sock(interface):
    try:
        with closing(socket.socket(family=socket.AF_INET6)) as sock:
            sock.bind((interface, 0))
    except OSError as sock_err:
        if sock_err.errno != errno.EADDRNOTAVAIL:
            raise
    else:
        return True
    return False
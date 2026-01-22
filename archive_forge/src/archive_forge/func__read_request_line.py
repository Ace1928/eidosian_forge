import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def _read_request_line(self):
    if self.rfile.closed:
        self.close_connection = 1
        return ''
    try:
        sock = self.connection
        if self.server.keepalive and (not isinstance(self.server.keepalive, bool)):
            sock.settimeout(self.server.keepalive)
        line = self.rfile.readline(self.server.url_length_limit)
        sock.settimeout(self.server.socket_timeout)
        return line
    except greenio.SSL.ZeroReturnError:
        pass
    except OSError as e:
        last_errno = support.get_errno(e)
        if last_errno in BROKEN_SOCK:
            self.server.log.debug('({}) connection reset by peer {!r}'.format(self.server.pid, self.client_address))
        elif last_errno not in BAD_SOCK:
            raise
    return ''
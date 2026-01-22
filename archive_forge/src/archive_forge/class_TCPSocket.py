import errno
import os
import socket
import ssl
import stat
import sys
import time
from gunicorn import util
class TCPSocket(BaseSocket):
    FAMILY = socket.AF_INET

    def __str__(self):
        if self.conf.is_ssl:
            scheme = 'https'
        else:
            scheme = 'http'
        addr = self.sock.getsockname()
        return '%s://%s:%d' % (scheme, addr[0], addr[1])

    def set_options(self, sock, bound=False):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return super().set_options(sock, bound=bound)
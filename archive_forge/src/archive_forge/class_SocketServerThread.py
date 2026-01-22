from __future__ import print_function
import logging
import os
import socket
import ssl
import sys
import threading
import warnings
from datetime import datetime
import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import trustme
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from urllib3.exceptions import HTTPWarning
from urllib3.util import ALPN_PROTOCOLS, resolve_cert_reqs, resolve_ssl_version
class SocketServerThread(threading.Thread):
    """
    :param socket_handler: Callable which receives a socket argument for one
        request.
    :param ready_event: Event which gets set when the socket handler is
        ready to receive requests.
    """
    USE_IPV6 = HAS_IPV6_AND_DNS

    def __init__(self, socket_handler, host='localhost', port=8081, ready_event=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.socket_handler = socket_handler
        self.host = host
        self.ready_event = ready_event

    def _start_server(self):
        if self.USE_IPV6:
            sock = socket.socket(socket.AF_INET6)
        else:
            warnings.warn('No IPv6 support. Falling back to IPv4.', NoIPv6Warning)
            sock = socket.socket(socket.AF_INET)
        if sys.platform != 'win32':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, 0))
        self.port = sock.getsockname()[1]
        sock.listen(1)
        if self.ready_event:
            self.ready_event.set()
        self.socket_handler(sock)
        sock.close()

    def run(self):
        self.server = self._start_server()
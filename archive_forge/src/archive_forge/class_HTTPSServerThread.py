from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import filter, str
from future import utils
import os
import sys
import ssl
import pprint
import socket
from future.backports.urllib import parse as urllib_parse
from future.backports.http.server import (HTTPServer as _HTTPServer,
from future.backports.test import support
class HTTPSServerThread(threading.Thread):

    def __init__(self, context, host=HOST, handler_class=None):
        self.flag = None
        self.server = HTTPSServer((host, 0), handler_class or RootedHTTPRequestHandler, context)
        self.port = self.server.server_port
        threading.Thread.__init__(self)
        self.daemon = True

    def __str__(self):
        return '<%s %s>' % (self.__class__.__name__, self.server)

    def start(self, flag=None):
        self.flag = flag
        threading.Thread.start(self)

    def run(self):
        if self.flag:
            self.flag.set()
        try:
            self.server.serve_forever(0.05)
        finally:
            self.server.server_close()

    def stop(self):
        self.server.shutdown()
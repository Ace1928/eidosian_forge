import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
class TCPConnectionHandler(socketserver.BaseRequestHandler):

    def handle(self):
        self.done = False
        self.handle_connection()
        while not self.done:
            self.handle_connection()

    def readline(self):
        req = self.request.recv(4096)
        if not req or (req.endswith(b'\n') and req.count(b'\n') == 1):
            return req
        raise ValueError('[{!r}] not a simple line'.format(req))

    def handle_connection(self):
        req = self.readline()
        if not req:
            self.done = True
        elif req == b'ping\n':
            self.request.sendall(b'pong\n')
        else:
            raise ValueError('[%s] not understood' % req)
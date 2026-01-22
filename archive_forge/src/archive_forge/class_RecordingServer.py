import io
import socket
import sys
import threading
from http.client import UnknownProtocol, parse_headers
from http.server import SimpleHTTPRequestHandler
import breezy
from .. import (config, controldir, debug, errors, osutils, tests, trace,
from ..bzr import remote as _mod_remote
from ..transport import remote
from ..transport.http import urllib
from ..transport.http.urllib import (AbstractAuthHandler, BasicAuthHandler,
from . import features, http_server, http_utils, test_server
from .scenarios import load_tests_apply_scenarios, multiply_scenarios
class RecordingServer:
    """A fake HTTP server.

    It records the bytes sent to it, and replies with a 200.
    """

    def __init__(self, expect_body_tail=None, scheme=''):
        """Constructor.

        :type expect_body_tail: str
        :param expect_body_tail: a reply won't be sent until this string is
            received.
        """
        self._expect_body_tail = expect_body_tail
        self.host = None
        self.port = None
        self.received_bytes = b''
        self.scheme = scheme

    def get_url(self):
        return '{}://{}:{}/'.format(self.scheme, self.host, self.port)

    def start_server(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(('127.0.0.1', 0))
        self.host, self.port = self._sock.getsockname()
        self._ready = threading.Event()
        self._thread = test_server.TestThread(sync_event=self._ready, target=self._accept_read_and_reply)
        self._thread.start()
        if 'threads' in tests.selftest_debug_flags:
            sys.stderr.write('Thread started: {}\n'.format(self._thread.ident))
        self._ready.wait()

    def _accept_read_and_reply(self):
        self._sock.listen(1)
        self._ready.set()
        conn, address = self._sock.accept()
        if self._expect_body_tail is not None:
            while not self.received_bytes.endswith(self._expect_body_tail):
                self.received_bytes += conn.recv(4096)
            conn.sendall(b'HTTP/1.1 200 OK\r\n')
        try:
            self._sock.close()
        except OSError:
            pass

    def stop_server(self):
        try:
            fake_conn = osutils.connect_socket((self.host, self.port))
            fake_conn.close()
        except OSError:
            pass
        self.host = None
        self.port = None
        self._thread.join()
        if 'threads' in tests.selftest_debug_flags:
            sys.stderr.write('Thread  joined: {}\n'.format(self._thread.ident))
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
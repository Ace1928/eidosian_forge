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
def test_post_body_is_received(self):
    server = RecordingServer(expect_body_tail=b'end-of-body', scheme=self._url_protocol)
    self.start_server(server)
    url = server.get_url()
    http_transport = transport.get_transport_from_url(url)
    code, response = http_transport._post(b'abc def end-of-body')
    self.assertTrue(server.received_bytes.startswith(b'POST /.bzr/smart HTTP/1.'))
    self.assertTrue(b'content-length: 19\r' in server.received_bytes.lower())
    self.assertTrue(b'content-type: application/octet-stream\r' in server.received_bytes.lower())
    self.assertTrue(server.received_bytes.endswith(b'\r\n\r\nabc def end-of-body'))
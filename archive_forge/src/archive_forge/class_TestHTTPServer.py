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
class TestHTTPServer(tests.TestCase):
    """Test the HTTP servers implementations."""

    def test_invalid_protocol(self):

        class BogusRequestHandler(http_server.TestingHTTPRequestHandler):
            protocol_version = 'HTTP/0.1'
        self.assertRaises(UnknownProtocol, http_server.HttpServer, BogusRequestHandler)

    def test_force_invalid_protocol(self):
        self.assertRaises(UnknownProtocol, http_server.HttpServer, protocol_version='HTTP/0.1')

    def test_server_start_and_stop(self):
        server = http_server.HttpServer()
        self.addCleanup(server.stop_server)
        server.start_server()
        self.assertTrue(server.server is not None)
        self.assertTrue(server.server.serving is not None)
        self.assertTrue(server.server.serving)

    def test_create_http_server_one_zero(self):

        class RequestHandlerOneZero(http_server.TestingHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
        server = http_server.HttpServer(RequestHandlerOneZero)
        self.start_server(server)
        self.assertIsInstance(server.server, http_server.TestingHTTPServer)

    def test_create_http_server_one_one(self):

        class RequestHandlerOneOne(http_server.TestingHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
        server = http_server.HttpServer(RequestHandlerOneOne)
        self.start_server(server)
        self.assertIsInstance(server.server, http_server.TestingThreadingHTTPServer)

    def test_create_http_server_force_one_one(self):

        class RequestHandlerOneZero(http_server.TestingHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
        server = http_server.HttpServer(RequestHandlerOneZero, protocol_version='HTTP/1.1')
        self.start_server(server)
        self.assertIsInstance(server.server, http_server.TestingThreadingHTTPServer)

    def test_create_http_server_force_one_zero(self):

        class RequestHandlerOneOne(http_server.TestingHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
        server = http_server.HttpServer(RequestHandlerOneOne, protocol_version='HTTP/1.0')
        self.start_server(server)
        self.assertIsInstance(server.server, http_server.TestingHTTPServer)
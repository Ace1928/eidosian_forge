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
class TestHttpTransportUrls(tests.TestCase):
    """Test the http urls."""
    scenarios = vary_by_http_client_implementation()

    def test_abs_url(self):
        """Construction of absolute http URLs"""
        t = self._transport('http://example.com/bzr/bzr.dev/')
        eq = self.assertEqualDiff
        eq(t.abspath('.'), 'http://example.com/bzr/bzr.dev')
        eq(t.abspath('foo/bar'), 'http://example.com/bzr/bzr.dev/foo/bar')
        eq(t.abspath('.bzr'), 'http://example.com/bzr/bzr.dev/.bzr')
        eq(t.abspath('.bzr/1//2/./3'), 'http://example.com/bzr/bzr.dev/.bzr/1/2/3')

    def test_invalid_http_urls(self):
        """Trap invalid construction of urls"""
        self._transport('http://example.com/bzr/bzr.dev/')
        self.assertRaises(urlutils.InvalidURL, self._transport, 'http://example.com:port/bzr/bzr.dev/')

    def test_http_root_urls(self):
        """Construction of URLs from server root"""
        t = self._transport('http://example.com/')
        eq = self.assertEqualDiff
        eq(t.abspath('.bzr/tree-version'), 'http://example.com/.bzr/tree-version')

    def test_http_impl_urls(self):
        """There are servers which ask for particular clients to connect"""
        server = self._server()
        server.start_server()
        try:
            url = server.get_url()
            self.assertTrue(url.startswith('%s://' % self._url_protocol))
        finally:
            server.stop_server()
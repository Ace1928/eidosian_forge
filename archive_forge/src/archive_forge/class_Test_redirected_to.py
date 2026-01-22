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
class Test_redirected_to(tests.TestCase):
    scenarios = vary_by_http_client_implementation()

    def test_redirected_to_subdir(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'http://www.example.com/foo/subdir')
        self.assertIsInstance(r, type(t))
        self.assertEqual(t._get_connection(), r._get_connection())
        self.assertEqual('http://www.example.com/foo/subdir/', r.base)

    def test_redirected_to_self_with_slash(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'http://www.example.com/foo/')
        self.assertIsInstance(r, type(t))
        self.assertEqual(t._get_connection(), r._get_connection())

    def test_redirected_to_host(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'http://foo.example.com/foo/subdir')
        self.assertIsInstance(r, type(t))
        self.assertEqual('http://foo.example.com/foo/subdir/', r.external_url())

    def test_redirected_to_same_host_sibling_protocol(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'https://www.example.com/foo')
        self.assertIsInstance(r, type(t))
        self.assertEqual('https://www.example.com/foo/', r.external_url())

    def test_redirected_to_same_host_different_protocol(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'bzr://www.example.com/foo')
        self.assertNotEqual(type(r), type(t))
        self.assertEqual('bzr://www.example.com/foo/', r.external_url())

    def test_redirected_to_same_host_specific_implementation(self):
        t = self._transport('http://www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'https+urllib://www.example.com/foo')
        self.assertEqual('https://www.example.com/foo/', r.external_url())

    def test_redirected_to_different_host_same_user(self):
        t = self._transport('http://joe@www.example.com/foo')
        r = t._redirected_to('http://www.example.com/foo', 'https://foo.example.com/foo')
        self.assertIsInstance(r, type(t))
        self.assertEqual(t._parsed_url.user, r._parsed_url.user)
        self.assertEqual('https://joe@foo.example.com/foo/', r.external_url())
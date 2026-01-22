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
class TestDoCatchRedirections(http_utils.TestCaseWithRedirectedWebserver):
    """Test transport.do_catching_redirections."""
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        self.build_tree_contents([('a', b'0123456789')])
        cleanup_http_redirection_connections(self)
        self.old_transport = self.get_old_transport()

    def get_a(self, t):
        return t.get('a')

    def test_no_redirection(self):
        t = self.get_new_transport()
        self.assertEqual(b'0123456789', transport.do_catching_redirections(self.get_a, t, None).read())

    def test_one_redirection(self):
        self.redirections = 0

        def redirected(t, exception, redirection_notice):
            self.redirections += 1
            redirected_t = t._redirected_to(exception.source, exception.target)
            return redirected_t
        self.assertEqual(b'0123456789', transport.do_catching_redirections(self.get_a, self.old_transport, redirected).read())
        self.assertEqual(1, self.redirections)

    def test_redirection_loop(self):

        def redirected(transport, exception, redirection_notice):
            return self.old_transport.clone(exception.target)
        self.assertRaises(errors.TooManyRedirections, transport.do_catching_redirections, self.get_a, self.old_transport, redirected)
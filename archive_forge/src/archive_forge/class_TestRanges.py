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
class TestRanges(http_utils.TestCaseWithWebserver):
    """Test the Range header in GET methods."""
    scenarios = multiply_scenarios(vary_by_http_client_implementation(), vary_by_http_protocol_version())

    def setUp(self):
        super().setUp()
        self.build_tree_contents([('a', b'0123456789')])

    def create_transport_readonly_server(self):
        return http_server.HttpServer(protocol_version=self._protocol_version)

    def _file_contents(self, relpath, ranges):
        t = self.get_readonly_transport()
        offsets = [(start, end - start + 1) for start, end in ranges]
        coalesce = t._coalesce_offsets
        coalesced = list(coalesce(offsets, limit=0, fudge_factor=0))
        code, data = t._get(relpath, coalesced)
        self.assertTrue(code in (200, 206), '_get returns: %d' % code)
        for start, end in ranges:
            data.seek(start)
            yield data.read(end - start + 1)

    def _file_tail(self, relpath, tail_amount):
        t = self.get_readonly_transport()
        code, data = t._get(relpath, [], tail_amount)
        self.assertTrue(code in (200, 206), '_get returns: %d' % code)
        data.seek(-tail_amount, 2)
        return data.read(tail_amount)

    def test_range_header(self):
        self.assertEqual([b'0', b'234'], list(self._file_contents('a', [(0, 0), (2, 4)])))

    def test_range_header_tail(self):
        self.assertEqual(b'789', self._file_tail('a', 3))

    def test_syntactically_invalid_range_header(self):
        self.assertListRaises(errors.InvalidHttpRange, self._file_contents, 'a', [(4, 3)])

    def test_semantically_invalid_range_header(self):
        self.assertListRaises(errors.InvalidHttpRange, self._file_contents, 'a', [(42, 128)])
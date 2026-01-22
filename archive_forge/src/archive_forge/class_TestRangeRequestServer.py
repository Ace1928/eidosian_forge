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
class TestRangeRequestServer(TestSpecificRequestHandler):
    """Tests readv requests against server.

    We test against default "normal" server.
    """

    def setUp(self):
        super().setUp()
        self.build_tree_contents([('a', b'0123456789')])

    def test_readv(self):
        t = self.get_readonly_transport()
        l = list(t.readv('a', ((0, 1), (1, 1), (3, 2), (9, 1))))
        self.assertEqual(l[0], (0, b'0'))
        self.assertEqual(l[1], (1, b'1'))
        self.assertEqual(l[2], (3, b'34'))
        self.assertEqual(l[3], (9, b'9'))

    def test_readv_out_of_order(self):
        t = self.get_readonly_transport()
        l = list(t.readv('a', ((1, 1), (9, 1), (0, 1), (3, 2))))
        self.assertEqual(l[0], (1, b'1'))
        self.assertEqual(l[1], (9, b'9'))
        self.assertEqual(l[2], (0, b'0'))
        self.assertEqual(l[3], (3, b'34'))

    def test_readv_invalid_ranges(self):
        t = self.get_readonly_transport()
        self.assertListRaises((errors.InvalidRange, errors.ShortReadvError), t.readv, 'a', [(1, 1), (8, 10)])
        self.assertListRaises((errors.InvalidRange, errors.ShortReadvError), t.readv, 'a', [(12, 2)])

    def test_readv_multiple_get_requests(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        t._max_readv_combine = 1
        t._max_get_ranges = 1
        l = list(t.readv('a', ((0, 1), (1, 1), (3, 2), (9, 1))))
        self.assertEqual(l[0], (0, b'0'))
        self.assertEqual(l[1], (1, b'1'))
        self.assertEqual(l[2], (3, b'34'))
        self.assertEqual(l[3], (9, b'9'))
        self.assertEqual(4, server.GET_request_nb)

    def test_readv_get_max_size(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        t._get_max_size = 2
        l = list(t.readv('a', ((0, 1), (1, 1), (2, 4), (6, 4))))
        self.assertEqual(l[0], (0, b'0'))
        self.assertEqual(l[1], (1, b'1'))
        self.assertEqual(l[2], (2, b'2345'))
        self.assertEqual(l[3], (6, b'6789'))
        self.assertEqual(3, server.GET_request_nb)

    def test_complete_readv_leave_pipe_clean(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        t._get_max_size = 2
        list(t.readv('a', ((0, 1), (1, 1), (2, 4), (6, 4))))
        self.assertEqual(3, server.GET_request_nb)
        self.assertEqual(b'0123456789', t.get_bytes('a'))
        self.assertEqual(4, server.GET_request_nb)

    def test_incomplete_readv_leave_pipe_clean(self):
        server = self.get_readonly_server()
        t = self.get_readonly_transport()
        t._get_max_size = 2
        ireadv = iter(t.readv('a', ((0, 1), (1, 1), (2, 4), (6, 4))))
        self.assertEqual((0, b'0'), next(ireadv))
        self.assertEqual(1, server.GET_request_nb)
        self.assertEqual(b'0123456789', t.get_bytes('a'))
        self.assertEqual(2, server.GET_request_nb)
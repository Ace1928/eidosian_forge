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
def test_readv_out_of_order(self):
    t = self.get_readonly_transport()
    l = list(t.readv('a', ((1, 1), (9, 1), (0, 1), (3, 2))))
    self.assertEqual(l[0], (1, b'1'))
    self.assertEqual(l[1], (9, b'9'))
    self.assertEqual(l[2], (0, b'0'))
    self.assertEqual(l[3], (3, b'34'))
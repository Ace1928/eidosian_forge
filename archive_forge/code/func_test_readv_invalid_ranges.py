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
def test_readv_invalid_ranges(self):
    t = self.get_readonly_transport()
    self.assertListRaises((errors.InvalidRange, errors.ShortReadvError), t.readv, 'a', [(1, 1), (8, 10)])
    self.assertListRaises((errors.InvalidRange, errors.ShortReadvError), t.readv, 'a', [(12, 2)])
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
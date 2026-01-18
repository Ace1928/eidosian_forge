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
def test_user_from_auth_conf(self):
    user = 'joe'
    password = 'foo'
    self.server.add_user(user, password)
    _setup_authentication_config(scheme='http', port=self.server.port, user=user, password=password)
    t = self.get_user_transport(None, None)
    with t.get('a') as f:
        self.assertEqual(b'contents of a\n', f.read())
    self.assertEqual(1, self.server.auth_required_errors)
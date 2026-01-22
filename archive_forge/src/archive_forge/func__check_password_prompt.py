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
def _check_password_prompt(self, scheme, user, actual_prompt):
    expected_prompt = self._password_prompt_prefix + "%s %s@%s:%d, Realm: '%s' password: " % (scheme.upper(), user, self.server.host, self.server.port, self.server.auth_realm)
    self.assertEqual(expected_prompt, actual_prompt)
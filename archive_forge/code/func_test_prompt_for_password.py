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
def test_prompt_for_password(self):
    self.server.add_user('joe', 'foo')
    t = self.get_user_transport('joe', None)
    ui.ui_factory = tests.TestUIFactory(stdin='foo\n')
    stdout, stderr = (ui.ui_factory.stdout, ui.ui_factory.stderr)
    self.assertEqual(b'contents of a\n', t.get('a').read())
    self.assertEqual('', ui.ui_factory.stdin.readline())
    self._check_password_prompt(t._unqualified_scheme, 'joe', stderr.getvalue())
    self.assertEqual('', stdout.getvalue())
    self.assertEqual(b'contents of b\n', t.get('b').read())
    t2 = t.clone()
    self.assertEqual(b'contents of b\n', t2.get('b').read())
    self.assertEqual(1, self.server.auth_required_errors)
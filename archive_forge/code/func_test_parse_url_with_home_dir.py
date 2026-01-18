import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def test_parse_url_with_home_dir(self):
    s = _mod_sftp.SFTPTransport('sftp://ro%62ey:h%40t@example.com:2222/~/relative')
    self.assertEqual(s._parsed_url.host, 'example.com')
    self.assertEqual(s._parsed_url.port, 2222)
    self.assertEqual(s._parsed_url.user, 'robey')
    self.assertEqual(s._parsed_url.password, 'h@t')
    self.assertEqual(s._parsed_url.path, '/~/relative/')
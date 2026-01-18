import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_parse_quoted_url(self):
    t = transport.ConnectedTransport('http://ro%62ey:h%40t@ex%41mple.com:2222/path')
    self.assertEqual(t._parsed_url.host, 'exAmple.com')
    self.assertEqual(t._parsed_url.port, 2222)
    self.assertEqual(t._parsed_url.user, 'robey')
    self.assertEqual(t._parsed_url.password, 'h@t')
    self.assertEqual(t._parsed_url.path, '/path/')
    self.assertEqual(t.base, 'http://ro%62ey@ex%41mple.com:2222/path/')
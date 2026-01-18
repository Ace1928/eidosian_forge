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
def test_parse_url_with_at_in_user(self):
    t = transport.ConnectedTransport('ftp://user@host.com@www.host.com/')
    self.assertEqual(t._parsed_url.user, 'user@host.com')
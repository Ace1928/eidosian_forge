import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_simple_pipes__accept_bytes_subprocess_closed(self):
    p = subprocess.Popen([sys.executable, '-c', 'import sys\nsys.stdout.write(sys.stdin.read(4))\nsys.stdout.close()\n'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
    client_medium = medium.SmartSimplePipesClientMedium(p.stdout, p.stdin, 'base')
    client_medium._accept_bytes(b'abc\n')
    self.assertEqual(b'abc', client_medium._read_bytes(3))
    p.wait()
    self.assertRaises(errors.ConnectionReset, client_medium._accept_bytes, b'more')
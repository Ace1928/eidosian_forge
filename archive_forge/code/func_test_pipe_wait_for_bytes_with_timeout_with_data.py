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
def test_pipe_wait_for_bytes_with_timeout_with_data(self):
    r_server, w_client = os.pipe()
    self.addCleanup(os.close, w_client)
    with os.fdopen(r_server, 'rb') as rf_server:
        server = self.create_pipe_medium(rf_server, None, None)
        os.write(w_client, b'data\n')
        server._wait_for_bytes_with_timeout(0.1)
        data = server.read_bytes(5)
        self.assertEqual(b'data\n', data)
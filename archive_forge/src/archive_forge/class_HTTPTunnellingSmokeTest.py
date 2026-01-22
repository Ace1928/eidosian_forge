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
class HTTPTunnellingSmokeTest(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideEnv('BRZ_NO_SMART_VFS', None)

    def test_smart_http_medium_request_accept_bytes(self):
        medium = FakeHTTPMedium()
        request = urllib.SmartClientHTTPMediumRequest(medium)
        request.accept_bytes(b'abc')
        request.accept_bytes(b'def')
        self.assertEqual(None, medium.written_request)
        request.finished_writing()
        self.assertEqual(b'abcdef', medium.written_request)
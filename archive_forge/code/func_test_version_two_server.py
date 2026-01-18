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
def test_version_two_server(self):
    """If the server only speaks protocol 2, the client will first try
        version 3, then fallback to protocol 2.

        Further, _SmartClient caches the detection, so future requests will all
        use protocol 2 immediately.
        """
    medium = MockMedium()
    smart_client = client._SmartClient(medium, headers={})
    medium.expect_request(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02de' + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', b'bzr response 2\nfailed\n\n')
    medium.expect_disconnect()
    medium.expect_request(b'bzr request 2\nmethod-name\x01arg 1\x01arg 2\n', b'bzr response 2\nsuccess\nresponse value\n')
    result = smart_client.call(b'method-name', b'arg 1', b'arg 2')
    self.assertEqual((b'response value',), result)
    medium.expect_request(b'bzr request 2\nanother-method\n', b'bzr response 2\nsuccess\nanother response\n')
    result = smart_client.call(b'another-method')
    self.assertEqual((b'another response',), result)
    self.assertEqual([], medium._expected_events)
    self.assertTrue(medium._is_remote_before((1, 6)))
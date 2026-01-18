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
def test_unknown_version(self):
    """If the server does not use any known (or at least supported)
        protocol version, a SmartProtocolError is raised.
        """
    medium = MockMedium()
    smart_client = client._SmartClient(medium, headers={})
    unknown_protocol_bytes = b'Unknown protocol!'
    medium.expect_request(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02de' + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', unknown_protocol_bytes)
    medium.expect_disconnect()
    medium.expect_request(b'bzr request 2\nmethod-name\x01arg 1\x01arg 2\n', unknown_protocol_bytes)
    medium.expect_disconnect()
    self.assertRaises(errors.SmartProtocolError, smart_client.call, b'method-name', b'arg 1', b'arg 2')
    self.assertEqual([], medium._expected_events)
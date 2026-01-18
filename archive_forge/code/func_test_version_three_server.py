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
def test_version_three_server(self):
    """With a protocol 3 server, only one request is needed."""
    medium = MockMedium()
    smart_client = client._SmartClient(medium, headers={})
    message_start = protocol.MESSAGE_VERSION_THREE + b'\x00\x00\x00\x02de'
    medium.expect_request(message_start + b's\x00\x00\x00\x1el11:method-name5:arg 15:arg 2ee', message_start + b's\x00\x00\x00\x13l14:response valueee')
    result = smart_client.call(b'method-name', b'arg 1', b'arg 2')
    self.assertEqual((b'response value',), result)
    self.assertEqual([], medium._expected_events)
    self.assertFalse(medium._is_remote_before((1, 6)))
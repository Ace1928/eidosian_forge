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
def test_repeated_excess(self):
    """Repeated calls to accept_bytes after the message end has been parsed
        accumlates the bytes in the unused_data attribute.
        """
    output = BytesIO()
    headers = b'\x00\x00\x00\x02de'
    end = b'e'
    request_bytes = headers + end
    smart_protocol = self.server_protocol_class(LoggingMessageHandler())
    smart_protocol.accept_bytes(request_bytes)
    self.assertEqual(b'', smart_protocol.unused_data)
    smart_protocol.accept_bytes(b'aaa')
    self.assertEqual(b'aaa', smart_protocol.unused_data)
    smart_protocol.accept_bytes(b'bbb')
    self.assertEqual(b'aaabbb', smart_protocol.unused_data)
    self.assertEqual(0, smart_protocol.next_read_size())
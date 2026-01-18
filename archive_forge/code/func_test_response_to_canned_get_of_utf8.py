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
def test_response_to_canned_get_of_utf8(self):
    transport = memory.MemoryTransport('memory:///')
    utf8_filename = 'testfile‽'.encode()
    hpss_path = urlutils.quote_from_bytes(utf8_filename)
    transport.put_bytes(hpss_path, b'contents\nof\nfile\n')
    server, from_server = self.create_pipe_context(b'get\x01' + hpss_path.encode('ascii') + b'\n', transport)
    smart_protocol = protocol.SmartServerRequestProtocolOne(transport, from_server.write)
    server._serve_one_request(smart_protocol)
    self.assertEqual(b'ok\n17\ncontents\nof\nfile\ndone\n', from_server.getvalue())
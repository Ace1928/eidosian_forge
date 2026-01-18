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
def test_socket_stream_with_bulk_data(self):
    sample_request_bytes = b'command\n9\nbulk datadone\n'
    server, client_sock = self.create_socket_context(None)
    sample_protocol = SampleRequest(expected_bytes=sample_request_bytes)
    client_sock.sendall(sample_request_bytes)
    server._serve_one_request(sample_protocol)
    server._disconnect_client()
    self.assertEqual(b'', client_sock.recv(1))
    self.assertEqual(sample_request_bytes, sample_protocol.accepted_bytes)
    self.assertFalse(server.finished)
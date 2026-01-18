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
def test_read_streamed_body_error(self):
    """When a stream is interrupted by an error..."""
    body_header = b'chunked\n'
    a_body_chunk = b'4\naaaa'
    err_signal = b'ERR\n'
    err_chunks = b'a\nerror arg1' + b'4\narg2'
    finish = b'END\n'
    body = body_header + a_body_chunk + err_signal + err_chunks + finish
    server_bytes = protocol.RESPONSE_VERSION_TWO + b'success\nok\n' + body
    input = BytesIO(server_bytes)
    output = BytesIO()
    client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
    smart_request = client_medium.get_request()
    smart_protocol = protocol.SmartClientRequestProtocolTwo(smart_request)
    smart_protocol.call(b'foo')
    smart_protocol.read_response_tuple(True)
    expected_chunks = [b'aaaa', _mod_request.FailedSmartServerResponse((b'error arg1', b'arg2'))]
    stream = smart_protocol.read_streamed_body()
    self.assertEqual(expected_chunks, list(stream))
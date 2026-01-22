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
class TestSmartClientStreamMediumRequest(tests.TestCase):
    """Tests the for SmartClientStreamMediumRequest.

    SmartClientStreamMediumRequest is a helper for the three stream based
    mediums: TCP, SSH, SimplePipes, so we only test it once, and then test that
    those three mediums implement the interface it expects.
    """

    def test_accept_bytes_after_finished_writing_errors(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        request.finished_writing()
        self.assertRaises(errors.WritingCompleted, request.accept_bytes, None)

    def test_accept_bytes(self):
        input = BytesIO()
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        request.accept_bytes(b'123')
        request.finished_writing()
        request.finished_reading()
        self.assertEqual(b'', input.getvalue())
        self.assertEqual(b'123', output.getvalue())

    def test_construct_sets_stream_request(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        self.assertIs(client_medium._current_request, request)

    def test_construct_while_another_request_active_throws(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        client_medium._current_request = 'a'
        self.assertRaises(medium.TooManyConcurrentRequests, medium.SmartClientStreamMediumRequest, client_medium)

    def test_finished_read_clears_current_request(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        request.finished_writing()
        request.finished_reading()
        self.assertEqual(None, client_medium._current_request)

    def test_finished_read_before_finished_write_errors(self):
        client_medium = medium.SmartSimplePipesClientMedium(None, None, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        self.assertRaises(errors.WritingNotComplete, request.finished_reading)

    def test_read_bytes(self):
        input = BytesIO(b'321')
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        request.finished_writing()
        self.assertEqual(b'321', request.read_bytes(3))
        request.finished_reading()
        self.assertEqual(b'', input.read())
        self.assertEqual(b'', output.getvalue())

    def test_read_bytes_before_finished_write_errors(self):
        client_medium = medium.SmartSimplePipesClientMedium(None, None, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        self.assertRaises(errors.WritingNotComplete, request.read_bytes, None)

    def test_read_bytes_after_finished_reading_errors(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = medium.SmartClientStreamMediumRequest(client_medium)
        request.finished_writing()
        request.finished_reading()
        self.assertRaises(errors.ReadingCompleted, request.read_bytes, None)

    def test_reset(self):
        server_sock, client_sock = portable_socket_pair()
        client_medium = medium.SmartTCPClientMedium(None, None, None)
        client_medium._socket = client_sock
        client_medium._connected = True
        req = client_medium.get_request()
        self.assertRaises(medium.TooManyConcurrentRequests, client_medium.get_request)
        client_medium.reset()
        self.assertFalse(client_medium._connected)
        self.assertIs(None, client_medium._socket)
        try:
            self.assertEqual('', client_sock.recv(1))
        except OSError as e:
            if e.errno not in (errno.EBADF,):
                raise
        req = client_medium.get_request()
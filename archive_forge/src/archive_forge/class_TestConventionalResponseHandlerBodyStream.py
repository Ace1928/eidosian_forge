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
class TestConventionalResponseHandlerBodyStream(tests.TestCase):

    def make_response_handler(self, response_bytes):
        from breezy.bzr.smart.message import ConventionalResponseHandler
        response_handler = ConventionalResponseHandler()
        protocol_decoder = protocol.ProtocolThreeDecoder(response_handler)
        protocol_decoder.state_accept = protocol_decoder._state_accept_expecting_message_part
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(BytesIO(response_bytes), output, 'base')
        medium_request = client_medium.get_request()
        medium_request.finished_writing()
        response_handler.setProtoAndMediumRequest(protocol_decoder, medium_request)
        return response_handler

    def test_interrupted_by_error(self):
        response_handler = self.make_response_handler(interrupted_body_stream)
        stream = response_handler.read_streamed_body()
        self.assertEqual(b'aaa', next(stream))
        self.assertEqual(b'bbb', next(stream))
        exc = self.assertRaises(errors.ErrorFromSmartServer, next, stream)
        self.assertEqual((b'error', b'Exception', b'Boom!'), exc.error_tuple)

    def test_interrupted_by_connection_lost(self):
        interrupted_body_stream = b'oSs\x00\x00\x00\x02leb\x00\x00\xff\xffincomplete chunk'
        response_handler = self.make_response_handler(interrupted_body_stream)
        stream = response_handler.read_streamed_body()
        self.assertRaises(errors.ConnectionReset, next, stream)

    def test_read_body_bytes_interrupted_by_connection_lost(self):
        interrupted_body_stream = b'oSs\x00\x00\x00\x02leb\x00\x00\xff\xffincomplete chunk'
        response_handler = self.make_response_handler(interrupted_body_stream)
        self.assertRaises(errors.ConnectionReset, response_handler.read_body_bytes)

    def test_multiple_bytes_parts(self):
        multiple_bytes_parts = b'oSs\x00\x00\x00\x02leb\x00\x00\x00\x0bSome bytes\nb\x00\x00\x00\nMore bytese'
        response_handler = self.make_response_handler(multiple_bytes_parts)
        self.assertEqual(b'Some bytes\nMore bytes', response_handler.read_body_bytes())
        response_handler = self.make_response_handler(multiple_bytes_parts)
        self.assertEqual([b'Some bytes\n', b'More bytes'], list(response_handler.read_streamed_body()))
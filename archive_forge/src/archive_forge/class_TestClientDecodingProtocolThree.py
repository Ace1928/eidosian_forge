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
class TestClientDecodingProtocolThree(TestSmartProtocol):
    """Tests for v3 of the client-side protocol decoding."""

    def make_logging_response_decoder(self):
        """Make v3 response decoder using a test response handler."""
        response_handler = LoggingMessageHandler()
        decoder = protocol.ProtocolThreeDecoder(response_handler)
        return (decoder, response_handler)

    def make_conventional_response_decoder(self):
        """Make v3 response decoder using a conventional response handler."""
        response_handler = message.ConventionalResponseHandler()
        decoder = protocol.ProtocolThreeDecoder(response_handler)
        response_handler.setProtoAndMediumRequest(decoder, StubRequest())
        return (decoder, response_handler)

    def test_trivial_response_decoding(self):
        """Smoke test for the simplest possible v3 response: empty headers,
        status byte, empty args, no body.
        """
        headers = b'\x00\x00\x00\x02de'
        response_status = b'oS'
        args = b's\x00\x00\x00\x02le'
        end = b'e'
        message_bytes = headers + response_status + args + end
        decoder, response_handler = self.make_logging_response_decoder()
        decoder.accept_bytes(message_bytes)
        self.assertEqual(0, decoder.next_read_size())
        self.assertEqual(b'', decoder.unused_data)
        self.assertEqual([('headers', {}), ('byte', b'S'), ('structure', ()), ('end',)], response_handler.event_log)

    def test_incomplete_message(self):
        """A decoder will keep signalling that it needs more bytes via
        next_read_size() != 0 until it has seen a complete message, regardless
        which state it is in.
        """
        headers = b'\x00\x00\x00\x02de'
        response_status = b'oS'
        args = b's\x00\x00\x00\x02le'
        body = b'b\x00\x00\x00\x04BODY'
        end = b'e'
        simple_response = headers + response_status + args + body + end
        decoder, response_handler = self.make_logging_response_decoder()
        for byte in bytearray(simple_response):
            self.assertNotEqual(0, decoder.next_read_size())
            decoder.accept_bytes(bytes([byte]))
        self.assertEqual(0, decoder.next_read_size())

    def test_read_response_tuple_raises_UnknownSmartMethod(self):
        """read_response_tuple raises UnknownSmartMethod if the server replied
        with 'UnknownMethod'.
        """
        headers = b'\x00\x00\x00\x02de'
        response_status = b'oE'
        args = b's\x00\x00\x00 l13:UnknownMethod11:method-namee'
        end = b'e'
        message_bytes = headers + response_status + args + end
        decoder, response_handler = self.make_conventional_response_decoder()
        decoder.accept_bytes(message_bytes)
        error = self.assertRaises(errors.UnknownSmartMethod, response_handler.read_response_tuple)
        self.assertEqual(b'method-name', error.verb)

    def test_read_response_tuple_error(self):
        """If the response has an error, it is raised as an exception."""
        headers = b'\x00\x00\x00\x02de'
        response_status = b'oE'
        args = b's\x00\x00\x00\x1al9:first arg10:second arge'
        end = b'e'
        message_bytes = headers + response_status + args + end
        decoder, response_handler = self.make_conventional_response_decoder()
        decoder.accept_bytes(message_bytes)
        error = self.assertRaises(errors.ErrorFromSmartServer, response_handler.read_response_tuple)
        self.assertEqual((b'first arg', b'second arg'), error.error_tuple)
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
class TestSmartProtocol(tests.TestCase):
    """Base class for smart protocol tests.

    Each test case gets a smart_server and smart_client created during setUp().

    It is planned that the client can be called with self.call_client() giving
    it an expected server response, which will be fed into it when it tries to
    read. Likewise, self.call_server will call a servers method with a canned
    serialised client request. Output done by the client or server for these
    calls will be captured to self.to_server and self.to_client. Each element
    in the list is a write call from the client or server respectively.

    Subclasses can override client_protocol_class and server_protocol_class.
    """
    request_encoder: object
    response_decoder: Type[protocol._StatefulDecoder]
    server_protocol_class: Type[protocol.SmartProtocolBase]
    client_protocol_class: Optional[Type[protocol.SmartProtocolBase]] = None

    def make_client_protocol_and_output(self, input_bytes=None):
        """
        :returns: a Request
        """
        if input_bytes is None:
            input = BytesIO()
        else:
            input = BytesIO(input_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        if self.client_protocol_class is not None:
            client_protocol = self.client_protocol_class(request)
            return (client_protocol, client_protocol, output)
        else:
            self.assertNotEqual(None, self.request_encoder)
            self.assertNotEqual(None, self.response_decoder)
            requester = self.request_encoder(request)
            response_handler = message.ConventionalResponseHandler()
            response_protocol = self.response_decoder(response_handler, expect_version_marker=True)
            response_handler.setProtoAndMediumRequest(response_protocol, request)
            return (requester, response_handler, output)

    def make_client_protocol(self, input_bytes=None):
        result = self.make_client_protocol_and_output(input_bytes=input_bytes)
        requester, response_handler, output = result
        return (requester, response_handler)

    def make_server_protocol(self):
        out_stream = BytesIO()
        smart_protocol = self.server_protocol_class(None, out_stream.write)
        return (smart_protocol, out_stream)

    def setUp(self):
        super().setUp()
        self.response_marker = getattr(self.client_protocol_class, 'response_marker', None)
        self.request_marker = getattr(self.client_protocol_class, 'request_marker', None)

    def assertOffsetSerialisation(self, expected_offsets, expected_serialised, requester):
        """Check that smart (de)serialises offsets as expected.

        We check both serialisation and deserialisation at the same time
        to ensure that the round tripping cannot skew: both directions should
        be as expected.

        :param expected_offsets: a readv offset list.
        :param expected_seralised: an expected serial form of the offsets.
        """
        readv_cmd = vfs.ReadvRequest(None, '/')
        offsets = readv_cmd._deserialise_offsets(expected_serialised)
        self.assertEqual(expected_offsets, offsets)
        serialised = requester._serialise_offsets(offsets)
        self.assertEqual(expected_serialised, serialised)

    def build_protocol_waiting_for_body(self):
        smart_protocol, out_stream = self.make_server_protocol()
        smart_protocol._has_dispatched = True
        smart_protocol.request = _mod_request.SmartServerRequestHandler(None, _mod_request.request_handlers, '/')

        class FakeCommand(_mod_request.SmartServerRequest):

            def do_body(self_cmd, body_bytes):
                self.end_received = True
                self.assertEqual(b'abcdefg', body_bytes)
                return _mod_request.SuccessfulSmartServerResponse((b'ok',))
        smart_protocol.request._command = FakeCommand(None)
        smart_protocol.accept_bytes(b'')
        return smart_protocol

    def assertServerToClientEncoding(self, expected_bytes, expected_tuple, input_tuples):
        """Assert that each input_tuple serialises as expected_bytes, and the
        bytes deserialise as expected_tuple.
        """
        for input_tuple in input_tuples:
            server_protocol, server_output = self.make_server_protocol()
            server_protocol._send_response(_mod_request.SuccessfulSmartServerResponse(input_tuple))
            self.assertEqual(expected_bytes, server_output.getvalue())
        requester, response_handler = self.make_client_protocol(expected_bytes)
        requester.call(b'foo')
        self.assertEqual(expected_tuple, response_handler.read_response_tuple())
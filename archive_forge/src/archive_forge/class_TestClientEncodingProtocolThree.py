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
class TestClientEncodingProtocolThree(TestSmartProtocol):
    request_encoder = protocol.ProtocolThreeRequester
    response_decoder = protocol.ProtocolThreeDecoder
    server_protocol_class = protocol.ProtocolThreeDecoder

    def make_client_encoder_and_output(self):
        result = self.make_client_protocol_and_output()
        requester, response_handler, output = result
        return (requester, output)

    def test_call_smoke_test(self):
        """A smoke test for ProtocolThreeRequester.call.

        This test checks that a particular simple invocation of call emits the
        correct bytes for that invocation.
        """
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({b'header name': b'header value'})
        requester.call(b'one arg')
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x1fd11:header name12:header valuees\x00\x00\x00\x0bl7:one argee', output.getvalue())

    def test_call_with_body_bytes_smoke_test(self):
        """A smoke test for ProtocolThreeRequester.call_with_body_bytes.

        This test checks that a particular simple invocation of
        call_with_body_bytes emits the correct bytes for that invocation.
        """
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({b'header name': b'header value'})
        requester.call_with_body_bytes((b'one arg',), b'body bytes')
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x1fd11:header name12:header valuees\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\nbody bytese', output.getvalue())

    def test_call_writes_just_once(self):
        """A bodyless request is written to the medium all at once."""
        medium_request = StubMediumRequest()
        encoder = protocol.ProtocolThreeRequester(medium_request)
        encoder.call(b'arg1', b'arg2', b'arg3')
        self.assertEqual(['accept_bytes', 'finished_writing'], medium_request.calls)

    def test_call_with_body_bytes_writes_just_once(self):
        """A request with body bytes is written to the medium all at once."""
        medium_request = StubMediumRequest()
        encoder = protocol.ProtocolThreeRequester(medium_request)
        encoder.call_with_body_bytes((b'arg', b'arg'), b'body bytes')
        self.assertEqual(['accept_bytes', 'finished_writing'], medium_request.calls)

    def test_call_with_body_stream_smoke_test(self):
        """A smoke test for ProtocolThreeRequester.call_with_body_stream.

        This test checks that a particular simple invocation of
        call_with_body_stream emits the correct bytes for that invocation.
        """
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({b'header name': b'header value'})
        stream = [b'chunk 1', b'chunk two']
        requester.call_with_body_stream((b'one arg',), stream)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x1fd11:header name12:header valuees\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\x07chunk 1b\x00\x00\x00\tchunk twoe', output.getvalue())

    def test_call_with_body_stream_empty_stream(self):
        """call_with_body_stream with an empty stream."""
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({})
        stream = []
        requester.call_with_body_stream((b'one arg',), stream)
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\x0bl7:one argee', output.getvalue())

    def test_call_with_body_stream_error(self):
        """call_with_body_stream will abort the streamed body with an
        error if the stream raises an error during iteration.

        The resulting request will still be a complete message.
        """
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({})

        def stream_that_fails():
            yield b'aaa'
            yield b'bbb'
            raise Exception('Boom!')
        self.assertRaises(Exception, requester.call_with_body_stream, (b'one arg',), stream_that_fails())
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\x03aaab\x00\x00\x00\x03bbboEs\x00\x00\x00\tl5:erroree', output.getvalue())

    def test_records_start_of_body_stream(self):
        requester, output = self.make_client_encoder_and_output()
        requester.set_headers({})
        in_stream = [False]

        def stream_checker():
            self.assertTrue(requester.body_stream_started)
            in_stream[0] = True
            yield b'content'
        flush_called = []
        orig_flush = requester.flush

        def tracked_flush():
            flush_called.append(in_stream[0])
            if in_stream[0]:
                self.assertTrue(requester.body_stream_started)
            else:
                self.assertFalse(requester.body_stream_started)
            return orig_flush()
        requester.flush = tracked_flush
        requester.call_with_body_stream((b'one arg',), stream_checker())
        self.assertEqual(b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02des\x00\x00\x00\x0bl7:one argeb\x00\x00\x00\x07contente', output.getvalue())
        self.assertEqual([False, True, True], flush_called)
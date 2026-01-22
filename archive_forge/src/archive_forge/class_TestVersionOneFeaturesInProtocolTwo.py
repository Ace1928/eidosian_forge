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
class TestVersionOneFeaturesInProtocolTwo(TestSmartProtocol, CommonSmartProtocolTestMixin):
    """Tests for version one smart protocol features as implemeted by version
    two.
    """
    client_protocol_class = protocol.SmartClientRequestProtocolTwo
    server_protocol_class = protocol.SmartServerRequestProtocolTwo

    def test_construct_version_two_server_protocol(self):
        smart_protocol = protocol.SmartServerRequestProtocolTwo(None, None)
        self.assertEqual(b'', smart_protocol.unused_data)
        self.assertEqual(b'', smart_protocol.in_buffer)
        self.assertFalse(smart_protocol._has_dispatched)
        self.assertEqual(1, smart_protocol.next_read_size())

    def test_construct_version_two_client_protocol(self):
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(None, output, 'base')
        request = client_medium.get_request()
        client_protocol = protocol.SmartClientRequestProtocolTwo(request)

    def test_accept_bytes_of_bad_request_to_protocol(self):
        out_stream = BytesIO()
        smart_protocol = self.server_protocol_class(None, out_stream.write)
        smart_protocol.accept_bytes(b'abc')
        self.assertEqual(b'abc', smart_protocol.in_buffer)
        smart_protocol.accept_bytes(b'\n')
        self.assertEqual(self.response_marker + b"failed\nerror\x01Generic bzr smart protocol error: bad request 'abc'\n", out_stream.getvalue())
        self.assertTrue(smart_protocol._has_dispatched)
        self.assertEqual(0, smart_protocol.next_read_size())

    def test_accept_body_bytes_to_protocol(self):
        protocol = self.build_protocol_waiting_for_body()
        self.assertEqual(6, protocol.next_read_size())
        protocol.accept_bytes(b'7\nabc')
        self.assertEqual(9, protocol.next_read_size())
        protocol.accept_bytes(b'defgd')
        protocol.accept_bytes(b'one\n')
        self.assertEqual(0, protocol.next_read_size())
        self.assertTrue(self.end_received)

    def test_accept_request_and_body_all_at_once(self):
        self.overrideEnv('BRZ_NO_SMART_VFS', None)
        mem_transport = memory.MemoryTransport()
        mem_transport.put_bytes('foo', b'abcdefghij')
        out_stream = BytesIO()
        smart_protocol = self.server_protocol_class(mem_transport, out_stream.write)
        smart_protocol.accept_bytes(b'readv\x01foo\n3\n3,3done\n')
        self.assertEqual(0, smart_protocol.next_read_size())
        self.assertEqual(self.response_marker + b'success\nreadv\n3\ndefdone\n', out_stream.getvalue())
        self.assertEqual(b'', smart_protocol.unused_data)
        self.assertEqual(b'', smart_protocol.in_buffer)

    def test_accept_excess_bytes_are_preserved(self):
        out_stream = BytesIO()
        smart_protocol = self.server_protocol_class(None, out_stream.write)
        smart_protocol.accept_bytes(b'hello\nhello\n')
        self.assertEqual(self.response_marker + b'success\nok\x012\n', out_stream.getvalue())
        self.assertEqual(b'hello\n', smart_protocol.unused_data)
        self.assertEqual(b'', smart_protocol.in_buffer)

    def test_accept_excess_bytes_after_body(self):
        server_protocol = self.build_protocol_waiting_for_body()
        server_protocol.accept_bytes(b'7\nabcdefgdone\n' + self.response_marker)
        self.assertTrue(self.end_received)
        self.assertEqual(self.response_marker, server_protocol.unused_data)
        self.assertEqual(b'', server_protocol.in_buffer)
        server_protocol.accept_bytes(b'Y')
        self.assertEqual(self.response_marker + b'Y', server_protocol.unused_data)
        self.assertEqual(b'', server_protocol.in_buffer)

    def test_accept_excess_bytes_after_dispatch(self):
        out_stream = BytesIO()
        smart_protocol = self.server_protocol_class(None, out_stream.write)
        smart_protocol.accept_bytes(b'hello\n')
        self.assertEqual(self.response_marker + b'success\nok\x012\n', out_stream.getvalue())
        smart_protocol.accept_bytes(self.request_marker + b'hel')
        self.assertEqual(self.request_marker + b'hel', smart_protocol.unused_data)
        smart_protocol.accept_bytes(b'lo\n')
        self.assertEqual(self.request_marker + b'hello\n', smart_protocol.unused_data)
        self.assertEqual(b'', smart_protocol.in_buffer)

    def test__send_response_sets_finished_reading(self):
        smart_protocol = self.server_protocol_class(None, lambda x: None)
        self.assertEqual(1, smart_protocol.next_read_size())
        smart_protocol._send_response(_mod_request.SuccessfulSmartServerResponse((b'x',)))
        self.assertEqual(0, smart_protocol.next_read_size())

    def test__send_response_errors_with_base_response(self):
        """Ensure that only the Successful/Failed subclasses are used."""
        smart_protocol = self.server_protocol_class(None, lambda x: None)
        self.assertRaises(AttributeError, smart_protocol._send_response, _mod_request.SmartServerResponse((b'x',)))

    def test_query_version(self):
        """query_version on a SmartClientProtocolTwo should return a number.

        The protocol provides the query_version because the domain level clients
        may all need to be able to probe for capabilities.
        """
        input = BytesIO(self.response_marker + b'success\nok\x012\n')
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        self.assertEqual(2, smart_protocol.query_version())

    def test_client_call_empty_response(self):
        self.assertServerToClientEncoding(self.response_marker + b'success\n\n', (b'',), [(), (b'',)])

    def test_client_call_three_element_response(self):
        self.assertServerToClientEncoding(self.response_marker + b'success\na\x01b\x0134\n', (b'a', b'b', b'34'), [(b'a', b'b', b'34')])

    def test_client_call_with_body_bytes_uploads(self):
        expected_bytes = self.request_marker + b'foo\n7\nabcdefgdone\n'
        input = BytesIO(b'\n')
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call_with_body_bytes((b'foo',), b'abcdefg')
        self.assertEqual(expected_bytes, output.getvalue())

    def test_client_call_with_body_readv_array(self):
        expected_bytes = self.request_marker + b'foo\n7\n1,2\n5,6done\n'
        input = BytesIO(b'\n')
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call_with_body_readv_array((b'foo',), [(1, 2), (5, 6)])
        self.assertEqual(expected_bytes, output.getvalue())

    def test_client_read_body_bytes_all(self):
        expected_bytes = b'1234567'
        server_bytes = self.response_marker + b'success\nok\n7\n1234567done\n'
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        self.assertEqual(expected_bytes, smart_protocol.read_body_bytes())

    def test_client_read_body_bytes_incremental(self):
        expected_bytes = b'1234567'
        server_bytes = self.response_marker + b'success\nok\n7\n1234567done\n'
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        self.assertEqual(expected_bytes[0:2], smart_protocol.read_body_bytes(2))
        self.assertEqual(expected_bytes[2:4], smart_protocol.read_body_bytes(2))
        self.assertEqual(expected_bytes[4:6], smart_protocol.read_body_bytes(2))
        self.assertEqual(expected_bytes[6:7], smart_protocol.read_body_bytes())

    def test_client_cancel_read_body_does_not_eat_body_bytes(self):
        server_bytes = self.response_marker + b'success\nok\n7\n1234567done\n'
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        smart_protocol.cancel_read_body()
        self.assertEqual(len(self.response_marker + b'success\nok\n'), input.tell())
        self.assertRaises(errors.ReadingCompleted, smart_protocol.read_body_bytes)

    def test_client_read_body_bytes_interrupted_connection(self):
        server_bytes = self.response_marker + b'success\nok\n999\nincomplete body'
        input = BytesIO(server_bytes)
        output = BytesIO()
        client_medium = medium.SmartSimplePipesClientMedium(input, output, 'base')
        request = client_medium.get_request()
        smart_protocol = self.client_protocol_class(request)
        smart_protocol.call(b'foo')
        smart_protocol.read_response_tuple(True)
        self.assertRaises(errors.ConnectionReset, smart_protocol.read_body_bytes)
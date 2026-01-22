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
class TestResponseEncodingProtocolThree(tests.TestCase):

    def make_response_encoder(self):
        out_stream = BytesIO()
        response_encoder = protocol.ProtocolThreeResponder(out_stream.write)
        return (response_encoder, out_stream)

    def test_send_error_unknown_method(self):
        encoder, out_stream = self.make_response_encoder()
        encoder.send_error(errors.UnknownSmartMethod('method name'))
        self.assertEndsWith(out_stream.getvalue(), b'oE' + b's\x00\x00\x00 l13:UnknownMethod11:method nameee')

    def test_send_broken_body_stream(self):
        encoder, out_stream = self.make_response_encoder()
        encoder._headers = {}

        def stream_that_fails():
            yield b'aaa'
            yield b'bbb'
            raise Exception('Boom!')
        response = _mod_request.SuccessfulSmartServerResponse((b'args',), body_stream=stream_that_fails())
        encoder.send_response(response)
        expected_response = b'bzr message 3 (bzr 1.6)\n\x00\x00\x00\x02de' + interrupted_body_stream
        self.assertEqual(expected_response, out_stream.getvalue())
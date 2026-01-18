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
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
class TestVersionOneFeaturesInProtocolThree(TestSmartProtocol, CommonSmartProtocolTestMixin):
    """Tests for version one smart protocol features as implemented by version
    three.
    """
    request_encoder = protocol.ProtocolThreeRequester
    response_decoder = protocol.ProtocolThreeDecoder
    server_protocol_class = staticmethod(protocol.build_server_protocol_three)

    def setUp(self):
        super().setUp()
        self.response_marker = protocol.MESSAGE_VERSION_THREE
        self.request_marker = protocol.MESSAGE_VERSION_THREE

    def test_construct_version_three_server_protocol(self):
        smart_protocol = protocol.ProtocolThreeDecoder(None)
        self.assertEqual(b'', smart_protocol.unused_data)
        self.assertEqual([], smart_protocol._in_buffer_list)
        self.assertEqual(0, smart_protocol._in_buffer_len)
        self.assertFalse(smart_protocol._has_dispatched)
        self.assertEqual(4, smart_protocol.next_read_size())
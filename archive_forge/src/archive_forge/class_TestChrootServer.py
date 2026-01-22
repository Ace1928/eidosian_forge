import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestChrootServer(tests.TestCase):

    def test_construct(self):
        backing_transport = memory.MemoryTransport()
        server = chroot.ChrootServer(backing_transport)
        self.assertEqual(backing_transport, server.backing_transport)

    def test_setUp(self):
        backing_transport = memory.MemoryTransport()
        server = chroot.ChrootServer(backing_transport)
        server.start_server()
        self.addCleanup(server.stop_server)
        self.assertTrue(server.scheme in transport._get_protocol_handlers().keys())

    def test_stop_server(self):
        backing_transport = memory.MemoryTransport()
        server = chroot.ChrootServer(backing_transport)
        server.start_server()
        server.stop_server()
        self.assertFalse(server.scheme in transport._get_protocol_handlers().keys())

    def test_get_url(self):
        backing_transport = memory.MemoryTransport()
        server = chroot.ChrootServer(backing_transport)
        server.start_server()
        self.addCleanup(server.stop_server)
        self.assertEqual('chroot-%d:///' % id(server), server.get_url())
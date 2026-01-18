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
def test_setUp(self):
    backing_transport = memory.MemoryTransport()
    server = chroot.ChrootServer(backing_transport)
    server.start_server()
    self.addCleanup(server.stop_server)
    self.assertTrue(server.scheme in transport._get_protocol_handlers().keys())
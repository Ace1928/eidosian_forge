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
def test_transport_fallback(self):
    """Transport with missing dependency causes no error"""
    saved_handlers = transport._get_protocol_handlers()
    self.addCleanup(transport._set_protocol_handlers, saved_handlers)
    transport._clear_protocol_handlers()
    transport.register_transport_proto('foo')
    transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BackupTransportHandler')
    transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BadTransportHandler')
    t = transport.get_transport_from_url('foo://fooserver/foo')
    self.assertTrue(isinstance(t, BackupTransportHandler))
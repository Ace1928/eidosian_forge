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
def test_transport_dependency(self):
    """Transport with missing dependency causes no error"""
    saved_handlers = transport._get_protocol_handlers()
    self.addCleanup(transport._set_protocol_handlers, saved_handlers)
    transport._clear_protocol_handlers()
    transport.register_transport_proto('foo')
    transport.register_lazy_transport('foo', 'breezy.tests.test_transport', 'BadTransportHandler')
    try:
        transport.get_transport_from_url('foo://fooserver/foo')
    except UnsupportedProtocol as e:
        self.assertEqual('Unsupported protocol for url "foo://fooserver/foo": Unable to import library "some_lib": testing missing dependency', str(e))
    else:
        self.fail('Did not raise UnsupportedProtocol')
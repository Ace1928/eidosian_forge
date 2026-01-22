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
class ReadonlyDecoratorTransportTest(tests.TestCase):
    """Readonly decoration specific tests."""

    def test_local_parameters(self):
        t = readonly.ReadonlyTransportDecorator('readonly+.')
        self.assertEqual(True, t.listable())
        self.assertEqual(True, t.is_readonly())

    def test_http_parameters(self):
        from breezy.tests.http_server import HttpServer
        server = HttpServer()
        self.start_server(server)
        t = transport.get_transport_from_url('readonly+' + server.get_url())
        self.assertIsInstance(t, readonly.ReadonlyTransportDecorator)
        self.assertEqual(False, t.listable())
        self.assertEqual(True, t.is_readonly())
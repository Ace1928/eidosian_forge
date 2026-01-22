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
class TestLocalTransports(tests.TestCase):

    def test_get_transport_from_abspath(self):
        here = osutils.abspath('.')
        t = transport.get_transport(here)
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base, urlutils.local_path_to_url(here) + '/')

    def test_get_transport_from_relpath(self):
        t = transport.get_transport('.')
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base, urlutils.local_path_to_url('.') + '/')

    def test_get_transport_from_local_url(self):
        here = osutils.abspath('.')
        here_url = urlutils.local_path_to_url(here) + '/'
        t = transport.get_transport(here_url)
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base, here_url)

    def test_local_abspath(self):
        here = osutils.abspath('.')
        t = transport.get_transport(here)
        self.assertEqual(t.local_abspath(''), here)
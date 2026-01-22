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
class TestTransportFromUrl(tests.TestCaseInTempDir):

    def test_with_path(self):
        self.assertRaises(urlutils.InvalidURL, transport.get_transport_from_url, self.test_dir)

    def test_with_url(self):
        url = urlutils.local_path_to_url(self.test_dir)
        t = transport.get_transport_from_url(url)
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base.rstrip('/'), url)

    def test_with_url_and_segment_parameters(self):
        url = urlutils.local_path_to_url(self.test_dir) + ',branch=foo'
        t = transport.get_transport_from_url(url)
        self.assertIsInstance(t, local.LocalTransport)
        self.assertEqual(t.base.rstrip('/'), url)
        with open(os.path.join(self.test_dir, 'afile'), 'w') as f:
            f.write('data')
        self.assertTrue(t.has('afile'))
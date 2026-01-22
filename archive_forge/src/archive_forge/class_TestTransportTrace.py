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
class TestTransportTrace(tests.TestCase):

    def test_decorator(self):
        t = transport.get_transport_from_url('trace+memory://')
        self.assertIsInstance(t, breezy.transport.trace.TransportTraceDecorator)

    def test_clone_preserves_activity(self):
        t = transport.get_transport_from_url('trace+memory://')
        t2 = t.clone('.')
        self.assertTrue(t is not t2)
        self.assertTrue(t._activity is t2._activity)

    def test_get(self):
        t = transport.get_transport_from_url('trace+memory:///')
        t.put_bytes('foo', b'barish')
        t.get('foo')
        expected_result = []
        expected_result.append(('put_bytes', 'foo', 6, None))
        expected_result.append(('get', 'foo'))
        self.assertEqual(expected_result, t._activity)

    def test_readv(self):
        t = transport.get_transport_from_url('trace+memory:///')
        t.put_bytes('foo', b'barish')
        list(t.readv('foo', [(0, 1), (3, 2)], adjust_for_latency=True, upper_limit=6))
        expected_result = []
        expected_result.append(('put_bytes', 'foo', 6, None))
        expected_result.append(('readv', 'foo', [(0, 1), (3, 2)], True, 6))
        self.assertEqual(expected_result, t._activity)
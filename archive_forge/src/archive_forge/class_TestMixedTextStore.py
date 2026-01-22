import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
class TestMixedTextStore(TestCaseInTempDir, TestStores):

    def get_store(self, path='.', compressed=True):
        t = transport.get_transport_from_path(path)
        return TextStore(t, compressed=compressed)

    def test_get_mixed(self):
        cs = self.get_store('.', compressed=True)
        s = self.get_store('.', compressed=False)
        cs.add(BytesIO(b'hello there'), b'a')
        self.assertPathExists('a.gz')
        self.assertFalse(os.path.lexists('a'))
        with gzip.GzipFile('a.gz') as f:
            self.assertEqual(f.read(), b'hello there')
        self.assertEqual(cs.has_id(b'a'), True)
        self.assertEqual(s.has_id(b'a'), True)
        self.assertEqual(cs.get(b'a').read(), b'hello there')
        self.assertEqual(s.get(b'a').read(), b'hello there')
        self.assertRaises(BzrError, s.add, BytesIO(b'goodbye'), b'a')
        s.add(BytesIO(b'goodbye'), b'b')
        self.assertPathExists('b')
        self.assertFalse(os.path.lexists('b.gz'))
        with open('b', 'rb') as f:
            self.assertEqual(f.read(), b'goodbye')
        self.assertEqual(cs.has_id(b'b'), True)
        self.assertEqual(s.has_id(b'b'), True)
        self.assertEqual(cs.get(b'b').read(), b'goodbye')
        self.assertEqual(s.get(b'b').read(), b'goodbye')
        self.assertRaises(BzrError, cs.add, BytesIO(b'again'), b'b')
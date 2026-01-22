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
class TestStores:
    """Mixin template class that provides some common tests for stores"""

    def check_content(self, store, fileid, value):
        with store.get(fileid) as f:
            self.assertEqual(f.read(), value)

    def fill_store(self, store):
        store.add(BytesIO(b'hello'), b'a')
        store.add(BytesIO(b'other'), b'b')
        store.add(BytesIO(b'something'), b'c')
        store.add(BytesIO(b'goodbye'), b'123123')

    def test_get(self):
        store = self.get_store()
        self.fill_store(store)
        self.check_content(store, b'a', b'hello')
        self.check_content(store, b'b', b'other')
        self.check_content(store, b'c', b'something')
        self.assertRaises(KeyError, self.check_content, store, b'd', None)

    def test_multiple_add(self):
        """Multiple add with same ID should raise a BzrError"""
        store = self.get_store()
        self.fill_store(store)
        self.assertRaises(BzrError, store.add, BytesIO(b'goodbye'), b'123123')
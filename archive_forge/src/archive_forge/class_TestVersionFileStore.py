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
class TestVersionFileStore(TestCaseWithTransport):

    def get_scope(self):
        return self._transaction

    def setUp(self):
        super().setUp()
        self.vfstore = VersionedFileStore(MemoryTransport(), versionedfile_class=WeaveFile)
        self.vfstore.get_scope = self.get_scope
        self._transaction = None

    def test_get_weave_registers_dirty_in_write(self):
        self._transaction = transactions.WriteTransaction()
        vf = self.vfstore.get_weave_or_empty(b'id', self._transaction)
        self._transaction.finish()
        self._transaction = None
        self.assertRaises(errors.OutSideTransaction, vf.add_lines, b'b', [], [])
        self._transaction = transactions.WriteTransaction()
        vf = self.vfstore.get_weave(b'id', self._transaction)
        self._transaction.finish()
        self._transaction = None
        self.assertRaises(errors.OutSideTransaction, vf.add_lines, b'b', [], [])

    def test_get_weave_readonly_cant_write(self):
        self._transaction = transactions.WriteTransaction()
        vf = self.vfstore.get_weave_or_empty(b'id', self._transaction)
        self._transaction.finish()
        self._transaction = transactions.ReadOnlyTransaction()
        vf = self.vfstore.get_weave_or_empty(b'id', self._transaction)
        self.assertRaises(errors.ReadOnlyError, vf.add_lines, b'b', [], [])

    def test___iter__escaped(self):
        self.vfstore = VersionedFileStore(MemoryTransport(), prefixed=True, escaped=True, versionedfile_class=WeaveFile)
        self.vfstore.get_scope = self.get_scope
        self._transaction = transactions.WriteTransaction()
        vf = self.vfstore.get_weave_or_empty(b' ', self._transaction)
        vf.add_lines(b'a', [], [])
        del vf
        self._transaction.finish()
        self.assertEqual([b' '], list(self.vfstore))
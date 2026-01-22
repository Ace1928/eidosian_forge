import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
class TestLockableFiles_LockDir(TestCaseInTempDir, _TestLockableFiles_mixin):
    """LockableFile tests run with LockDir underneath"""

    def setUp(self):
        super().setUp()
        self.transport = transport.get_transport_from_path('.')
        self.lockable = self.get_lockable()
        self.lockable.create_lock()

    def get_lockable(self):
        return LockableFiles(self.transport, 'my-lock', lockdir.LockDir)

    def test_lock_created(self):
        self.assertTrue(self.transport.has('my-lock'))
        self.lockable.lock_write()
        self.assertTrue(self.transport.has('my-lock/held/info'))
        self.lockable.unlock()
        self.assertFalse(self.transport.has('my-lock/held/info'))
        self.assertTrue(self.transport.has('my-lock'))

    def test__file_modes(self):
        self.transport.mkdir('readonly')
        osutils.make_readonly('readonly')
        lockable = LockableFiles(self.transport.clone('readonly'), 'test-lock', lockdir.LockDir)
        self.assertEqual(448, lockable._dir_mode & 448)
        self.assertEqual(384, lockable._file_mode & 448)
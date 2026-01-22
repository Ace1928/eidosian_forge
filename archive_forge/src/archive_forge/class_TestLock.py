from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
class TestLock(TestCaseWithLock):

    def setUp(self):
        super().setUp()
        self.build_tree(['a-file'])

    def test_read_lock(self):
        """Smoke test for read locks."""
        a_lock = self.read_lock('a-file')
        self.addCleanup(a_lock.unlock)
        txt = a_lock.f.read()
        self.assertEqual(b'contents of a-file\n', txt)

    def test_create_if_needed_read(self):
        """We will create the file if it doesn't exist yet."""
        a_lock = self.read_lock('other-file')
        self.addCleanup(a_lock.unlock)
        txt = a_lock.f.read()
        self.assertEqual(b'', txt)

    def test_create_if_needed_write(self):
        """We will create the file if it doesn't exist yet."""
        a_lock = self.write_lock('other-file')
        self.addCleanup(a_lock.unlock)
        txt = a_lock.f.read()
        self.assertEqual(b'', txt)
        a_lock.f.seek(0)
        a_lock.f.write(b'foo\n')
        a_lock.f.seek(0)
        txt = a_lock.f.read()
        self.assertEqual(b'foo\n', txt)

    def test_readonly_file(self):
        """If the file is readonly, we can take a read lock.

        But we shouldn't be able to take a write lock.
        """
        self.requireFeature(features.not_running_as_root)
        osutils.make_readonly('a-file')
        self.assertRaises(IOError, open, 'a-file', 'rb+')
        a_lock = self.read_lock('a-file')
        a_lock.unlock()
        self.assertRaises(errors.LockFailed, self.write_lock, 'a-file')

    def test_write_lock(self):
        """Smoke test for write locks."""
        a_lock = self.write_lock('a-file')
        self.addCleanup(a_lock.unlock)
        txt = a_lock.f.read()
        self.assertEqual(b'contents of a-file\n', txt)
        a_lock.f.seek(0, 2)
        a_lock.f.write(b'more content\n')
        a_lock.f.seek(0)
        txt = a_lock.f.read()
        self.assertEqual(b'contents of a-file\nmore content\n', txt)

    def test_multiple_read_locks(self):
        """You can take out more than one read lock on the same file."""
        a_lock = self.read_lock('a-file')
        self.addCleanup(a_lock.unlock)
        b_lock = self.read_lock('a-file')
        self.addCleanup(b_lock.unlock)

    def test_multiple_write_locks_exclude(self):
        """Taking out more than one write lock should fail."""
        a_lock = self.write_lock('a-file')
        self.addCleanup(a_lock.unlock)
        self.assertRaises(errors.LockContention, self.write_lock, 'a-file')

    def _disabled_test_read_then_write_excludes(self):
        """If a file is read-locked, taking out a write lock should fail."""
        a_lock = self.read_lock('a-file')
        self.addCleanup(a_lock.unlock)
        self.assertRaises(errors.LockContention, self.write_lock, 'a-file')

    def test_read_unlock_write(self):
        """Make sure that unlocking allows us to lock write"""
        a_lock = self.read_lock('a-file')
        a_lock.unlock()
        a_lock = self.write_lock('a-file')
        a_lock.unlock()

    def _disabled_test_write_then_read_excludes(self):
        """If a file is write-locked, taking out a read lock should fail.

        The file is exclusively owned by the write lock, so we shouldn't be
        able to take out a shared read lock.
        """
        a_lock = self.write_lock('a-file')
        self.addCleanup(a_lock.unlock)
        self.assertRaises(errors.LockContention, self.read_lock, 'a-file')

    def _disabled_test_write_unlock_read(self):
        """If we have removed the write lock, we can grab a read lock."""
        a_lock = self.write_lock('a-file')
        a_lock.unlock()
        a_lock = self.read_lock('a-file')
        a_lock.unlock()

    def _disabled_test_multiple_read_unlock_write(self):
        """We can only grab a write lock if all read locks are done."""
        a_lock = b_lock = c_lock = None
        try:
            a_lock = self.read_lock('a-file')
            b_lock = self.read_lock('a-file')
            self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
            a_lock.unlock()
            a_lock = None
            self.assertRaises(errors.LockContention, self.write_lock, 'a-file')
            b_lock.unlock()
            b_lock = None
            c_lock = self.write_lock('a-file')
            c_lock.unlock()
            c_lock = None
        finally:
            if a_lock is not None:
                a_lock.unlock()
            if b_lock is not None:
                b_lock.unlock()
            if c_lock is not None:
                c_lock.unlock()
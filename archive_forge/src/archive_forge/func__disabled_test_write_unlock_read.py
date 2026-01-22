from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def _disabled_test_write_unlock_read(self):
    """If we have removed the write lock, we can grab a read lock."""
    a_lock = self.write_lock('a-file')
    a_lock.unlock()
    a_lock = self.read_lock('a-file')
    a_lock.unlock()
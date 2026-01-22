from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
class TestLockUnicodePath(TestCaseWithLock):
    _test_needs_features = [features.UnicodeFilenameFeature]

    def test_read_lock(self):
        self.build_tree(['ሴ'])
        u_lock = self.read_lock('ሴ')
        self.addCleanup(u_lock.unlock)

    def test_write_lock(self):
        self.build_tree(['ሴ'])
        u_lock = self.write_lock('ሴ')
        self.addCleanup(u_lock.unlock)
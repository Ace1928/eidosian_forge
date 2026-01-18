from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
def test_write_lock(self):
    self.build_tree(['ሴ'])
    u_lock = self.write_lock('ሴ')
    self.addCleanup(u_lock.unlock)
from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def assertRemovedAndDeleted(self, files):
    self.assertNotInWorkingTree(files)
    self.assertPathDoesNotExist(files)
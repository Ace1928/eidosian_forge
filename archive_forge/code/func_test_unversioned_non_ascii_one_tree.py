from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversioned_non_ascii_one_tree(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    if not isinstance(tree, InventoryTree):
        raise TestNotApplicable('test not applicable on non-inventory tests')
    self.build_tree(['§'])
    self.assertExpectedIds([], tree, ['§'], require_versioned=False)
    self.addCleanup(tree.lock_read().unlock)
    e = self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, ['§'])
    self.assertEqual(['§'], e.paths)
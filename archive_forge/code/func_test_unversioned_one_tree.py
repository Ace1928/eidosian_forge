from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversioned_one_tree(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryTree):
        raise TestNotApplicable('test not applicable on non-inventory tests')
    self.build_tree(['tree/unversioned'])
    self.assertExpectedIds([], tree, ['unversioned'], require_versioned=False)
    tree.lock_read()
    self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, ['unversioned'])
    tree.unlock()
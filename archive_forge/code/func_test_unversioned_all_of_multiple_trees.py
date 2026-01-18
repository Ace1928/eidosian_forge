from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversioned_all_of_multiple_trees(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryTree):
        raise TestNotApplicable('test not applicable on non-inventory tests')
    tree.commit('make basis')
    basis = tree.basis_tree()
    self.assertExpectedIds([], tree, ['unversioned'], [basis], require_versioned=False)
    tree.lock_read()
    basis.lock_read()
    self.assertRaises(errors.PathsNotVersionedError, tree.paths2ids, ['unversioned'], [basis])
    self.assertRaises(errors.PathsNotVersionedError, basis.paths2ids, ['unversioned'], [tree])
    basis.unlock()
    tree.unlock()
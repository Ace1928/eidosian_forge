from breezy import errors
from breezy.bzr import inventory
from breezy.bzr.workingtree import InventoryModified, InventoryWorkingTree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_trivial_read(self):
    tree = self.make_branch_and_tree('t1')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('read_working_inventory not usable on non-inventory working trees')
    tree.lock_read()
    self.assertIsInstance(tree.read_working_inventory(), inventory.Inventory)
    tree.unlock()
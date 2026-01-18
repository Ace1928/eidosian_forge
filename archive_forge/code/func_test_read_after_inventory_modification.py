from breezy import errors
from breezy.bzr import inventory
from breezy.bzr.workingtree import InventoryModified, InventoryWorkingTree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_read_after_inventory_modification(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('read_working_inventory not usable on non-inventory working trees')
    with tree.lock_write():
        tree.set_root_id(b'new-root')
        try:
            tree.read_working_inventory()
        except InventoryModified:
            pass
        else:
            self.assertEqual(b'new-root', tree.path2id(''))
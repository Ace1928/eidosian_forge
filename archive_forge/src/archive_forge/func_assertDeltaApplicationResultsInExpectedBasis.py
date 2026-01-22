import os
from io import BytesIO
from ... import errors
from ... import revision as _mod_revision
from ...bzr.inventory import (Inventory, InventoryDirectory, InventoryFile,
from ...bzr.inventorytree import InventoryRevisionTree, InventoryTree
from ...tests import TestNotApplicable
from ...uncommit import uncommit
from .. import features
from ..per_workingtree import TestCaseWithWorkingTree
def assertDeltaApplicationResultsInExpectedBasis(self, tree, revid, delta, expected_inventory):
    with tree.lock_write():
        tree.update_basis_by_delta(revid, delta)
    self.assertEqual(revid, tree.last_revision())
    self.assertEqual([revid], tree.get_parent_ids())
    result_basis = tree.basis_tree()
    with result_basis.lock_read():
        self.assertEqual(expected_inventory, result_basis.root_inventory)
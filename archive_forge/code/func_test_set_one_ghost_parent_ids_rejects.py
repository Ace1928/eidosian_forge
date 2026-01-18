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
def test_set_one_ghost_parent_ids_rejects(self):
    t = self.make_branch_and_tree('.')
    self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [b'missing-revision-id'])
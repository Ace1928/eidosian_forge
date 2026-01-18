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
def test_set_duplicate_parent_ids(self):
    t = self.make_branch_and_tree('.')
    rev1 = t.commit('first post')
    uncommit(t.branch, tree=t)
    rev2 = t.commit('second post')
    uncommit(t.branch, tree=t)
    rev3 = t.commit('third post')
    uncommit(t.branch, tree=t)
    t.set_parent_ids([rev1, rev2, rev2, rev3])
    self.assertConsistentParents([rev1, rev2, rev3], t)
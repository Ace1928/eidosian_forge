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
def test_set_two_parents_one_ghost_ids(self):
    t = self.make_branch_and_tree('.')
    revision_in_repo = t.commit('first post')
    uncommit(t.branch, tree=t)
    rev_tree = t.branch.repository.revision_tree(revision_in_repo)
    if t._format.supports_righthand_parent_id_as_ghost:
        t.set_parent_ids([revision_in_repo, b'another-missing'])
        self.assertConsistentParents([revision_in_repo, b'another-missing'], t)
    else:
        self.assertRaises(errors.GhostRevisionUnusableHere, t.set_parent_ids, [revision_in_repo, b'another-missing'])
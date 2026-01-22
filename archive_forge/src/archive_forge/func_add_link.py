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
def add_link(self, inv, rev_id, file_id, parent_id, name, target):
    new_link = InventoryLink(file_id, name, parent_id)
    new_link.symlink_target = target
    self.add_entry(inv, rev_id, new_link)
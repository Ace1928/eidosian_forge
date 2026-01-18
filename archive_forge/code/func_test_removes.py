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
def test_removes(self):
    old_revid = b'old-parent'
    basis_shape = Inventory(root_id=None)
    self.add_dir(basis_shape, old_revid, b'root-id', None, '')
    self.add_dir(basis_shape, old_revid, b'dir-id-A', b'root-id', 'A')
    self.add_link(basis_shape, old_revid, b'link-id-B', b'root-id', 'B', 'C')
    self.add_file(basis_shape, old_revid, b'file-id-C', b'root-id', 'C', b'1' * 32, 12)
    self.add_file(basis_shape, old_revid, b'file-id-D', b'dir-id-A', 'D', b'2' * 32, 24)
    new_revid = b'new-parent'
    new_shape = Inventory(root_id=None)
    self.add_new_root(new_shape, old_revid, new_revid)
    self.assertTransitionFromBasisToShape(basis_shape, old_revid, new_shape, new_revid)
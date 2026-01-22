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
class ShapeTree(InventoryRevisionTree):

    def __init__(self, shape):
        self._repository = tree.branch.repository
        self._inventory = shape

    def get_file_text(self, path):
        file_id = self.path2id(path)
        ie = self.root_inventory.get_entry(file_id)
        if ie.kind != 'file':
            return b''
        return b'a' * ie.text_size

    def get_file(self, path):
        return BytesIO(self.get_file_text(path))
import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def unchanged(self, tree, path):
    entry = self.get_path_entry(tree, path)
    parent = entry.parent_id
    name = entry.name
    kind = entry.kind
    executable = entry.executable
    return InventoryTreeChange(entry.file_id, (path, path), False, (True, True), (parent, parent), (name, name), (kind, kind), (executable, executable))
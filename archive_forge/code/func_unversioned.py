import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def unversioned(self, tree, path):
    """Create an unversioned result."""
    _, basename = os.path.split(path)
    kind = tree._comparison_data(None, path)[0]
    return InventoryTreeChange(None, (None, path), True, (False, False), (None, None), (None, basename), (None, kind), (None, False))
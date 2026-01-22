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
def assertConsistentParents(self, expected, tree):
    """Check that the parents found are as expected.

        This test helper also checks that they are consistent with
        the pre-get_parent_ids() api - which is now deprecated.
        """
    self.assertEqual(expected, tree.get_parent_ids())
    if expected == []:
        self.assertEqual(_mod_revision.NULL_REVISION, tree.last_revision())
    else:
        self.assertEqual(expected[0], tree.last_revision())
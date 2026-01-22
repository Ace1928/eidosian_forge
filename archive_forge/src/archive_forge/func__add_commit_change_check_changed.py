import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _add_commit_change_check_changed(self, tree, names, changer, expect_fs_hash=False):
    tree.add([names[0]])
    self.assertTrue(tree.is_versioned(names[0]))
    self._commit_change_check_changed(tree, names, changer, expect_fs_hash=expect_fs_hash)
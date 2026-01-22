import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _get_revtrees(self, tree, revision_ids):
    with tree.lock_read():
        trees = list(tree.branch.repository.revision_trees(revision_ids))
        for _tree in trees:
            _tree.lock_read()
            self.addCleanup(_tree.unlock)
        return trees
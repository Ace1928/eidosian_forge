import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _commit_sprout_rename_merge(self, tree1, name, expect_fs_hash=False):
    """Do a rename in both trees."""
    rev1, tree2 = self._commit_sprout(tree1, name)
    if tree2.supports_file_ids:
        file_id = tree2.path2id(name)
        self.assertIsNot(None, file_id)
    self.assertTrue(tree2.is_versioned(name))
    rev2 = self._rename_in_tree(tree1, name, 'rev2')
    rev3 = self._rename_in_tree(tree2, name, 'rev3')
    tree1.merge_from_branch(tree2.branch)
    rev4 = self.mini_commit_record_iter_changes(tree1, 'new_' + name, 'new_' + name, expect_fs_hash=expect_fs_hash, delta_against_basis=tree1.supports_rename_tracking())
    tree3, = self._get_revtrees(tree1, [rev4])
    if tree1.supports_file_ids:
        expected_graph = {}
        if tree1.supports_rename_tracking():
            self.assertEqual(rev4, tree3.get_file_revision('new_' + name))
            expected_graph[file_id, rev1] = ()
            expected_graph[file_id, rev2] = ((file_id, rev1),)
            expected_graph[file_id, rev3] = ((file_id, rev1),)
            expected_graph[file_id, rev4] = ((file_id, rev2), (file_id, rev3))
        else:
            self.assertEqual(rev2, tree3.get_file_revision('new_' + name))
            expected_graph[file_id, rev4] = ()
        self.assertFileGraph(expected_graph, tree1, (file_id, rev4))
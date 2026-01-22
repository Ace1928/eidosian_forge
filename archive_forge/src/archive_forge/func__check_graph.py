import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def _check_graph(in_tree, changed_in_tree):
    rev3 = self.mini_commit_record_iter_changes(in_tree, name, 'new_' + name, False, delta_against_basis=changed_in_tree)
    tree3, = self._get_revtrees(in_tree, [rev2])
    self.assertEqual(rev2, tree3.get_file_revision('new_' + name))
    expected_graph = {}
    expected_graph[file_id, rev1] = ()
    expected_graph[file_id, rev2] = ((file_id, rev1),)
    self.assertFileGraph(expected_graph, in_tree, (file_id, rev2))
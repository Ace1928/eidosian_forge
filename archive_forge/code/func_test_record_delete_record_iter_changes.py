import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_record_delete_record_iter_changes(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add(['foo'])
    foo_id = tree.path2id('foo')
    rev_id = tree.commit('added foo')
    with tree.lock_write():
        builder = tree.branch.get_commit_builder([rev_id])
        try:
            delete_change = InventoryTreeChange(foo_id, ('foo', None), True, (True, False), (tree.path2id(''), None), ('foo', None), ('file', None), (False, None))
            list(builder.record_iter_changes(tree, rev_id, [delete_change]))
            self.assertEqual(('foo', None, foo_id, None), builder.get_basis_delta()[0])
            self.assertTrue(builder.any_changes())
            builder.finish_inventory()
            rev_id2 = builder.commit('delete foo')
        except:
            builder.abort()
            raise
    rev_tree = builder.revision_tree()
    rev_tree.lock_read()
    self.addCleanup(rev_tree.unlock)
    self.assertFalse(rev_tree.is_versioned('foo'))
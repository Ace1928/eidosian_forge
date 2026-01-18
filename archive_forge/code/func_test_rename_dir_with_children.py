import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_dir_with_children(self):
    wt = self.make_branch_and_tree('.')
    wt.lock_write()
    root_id = wt.path2id('')
    self.addCleanup(wt.unlock)
    self.build_tree(['foo/', 'foo/bar'])
    wt.add(['foo', 'foo/bar'], ids=[b'foo-id', b'bar-id'])
    wt.apply_inventory_delta([('foo', 'baz', b'foo-id', inventory.InventoryDirectory(b'foo-id', 'baz', root_id))])
    self.assertEqual('baz', wt.id2path(b'foo-id'))
    self.assertEqual('baz/bar', wt.id2path(b'bar-id'))
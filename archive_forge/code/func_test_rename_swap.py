import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_swap(self):
    """Test the swap-names edge case.

        foo and bar should swap names, but retain their children.  If this
        works, any simpler rename ought to work.
        """
    wt = self.make_branch_and_tree('.')
    wt.lock_write()
    root_id = wt.path2id('')
    self.addCleanup(wt.unlock)
    self.build_tree(['foo/', 'foo/bar', 'baz/', 'baz/qux'])
    wt.add(['foo', 'foo/bar', 'baz', 'baz/qux'], ids=[b'foo-id', b'bar-id', b'baz-id', b'qux-id'])
    wt.apply_inventory_delta([('foo', 'baz', b'foo-id', inventory.InventoryDirectory(b'foo-id', 'baz', root_id)), ('baz', 'foo', b'baz-id', inventory.InventoryDirectory(b'baz-id', 'foo', root_id))])
    self.assertEqual('baz/bar', wt.id2path(b'bar-id'))
    self.assertEqual('foo/qux', wt.id2path(b'qux-id'))
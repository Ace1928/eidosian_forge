import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestApplyInventoryDelta(TestCaseWithWorkingTree):

    def setUp(self):
        super().setUp()
        if not self.bzrdir_format.repository_format.supports_full_versioned_files:
            raise tests.TestNotApplicable('format does not support inventory deltas')

    def test_add(self):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        root_id = wt.path2id('')
        wt.apply_inventory_delta([(None, 'bar/foo', b'foo-id', inventory.InventoryFile(b'foo-id', 'foo', parent_id=b'bar-id')), (None, 'bar', b'bar-id', inventory.InventoryDirectory(b'bar-id', 'bar', parent_id=root_id))])
        self.assertEqual('bar/foo', wt.id2path(b'foo-id'))
        self.assertEqual('bar', wt.id2path(b'bar-id'))

    def test_remove(self):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        self.build_tree(['foo/', 'foo/bar'])
        wt.add(['foo', 'foo/bar'], ids=[b'foo-id', b'bar-id'])
        wt.apply_inventory_delta([('foo', None, b'foo-id', None), ('foo/bar', None, b'bar-id', None)])
        self.assertFalse(wt.is_versioned('foo'))

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

    def test_rename_dir_with_children_with_children(self):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        root_id = wt.path2id('')
        self.addCleanup(wt.unlock)
        self.build_tree(['foo/', 'foo/bar/', 'foo/bar/baz'])
        wt.add(['foo', 'foo/bar', 'foo/bar/baz'], ids=[b'foo-id', b'bar-id', b'baz-id'])
        wt.apply_inventory_delta([('foo', 'quux', b'foo-id', inventory.InventoryDirectory(b'foo-id', 'quux', root_id))])
        self.assertEqual('quux/bar/baz', wt.id2path(b'baz-id'))

    def test_rename_file(self):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        self.build_tree(['foo/', 'foo/bar', 'baz/'])
        wt.add(['foo', 'foo/bar', 'baz'], ids=[b'foo-id', b'bar-id', b'baz-id'])
        wt.apply_inventory_delta([('foo/bar', 'baz/bar', b'bar-id', inventory.InventoryFile(b'bar-id', 'bar', b'baz-id'))])
        self.assertEqual('baz/bar', wt.id2path(b'bar-id'))

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

    def test_child_rename_ordering(self):
        """Test the rename-parent, move child edge case.

        (A naive implementation may move the parent first, and then be
         unable to find the child.)
        """
        wt = self.make_branch_and_tree('.')
        root_id = wt.path2id('')
        self.build_tree(['dir/', 'dir/child', 'other/'])
        wt.add(['dir', 'dir/child', 'other'], ids=[b'dir-id', b'child-id', b'other-id'])
        wt.apply_inventory_delta([('dir', 'dir2', b'dir-id', inventory.InventoryDirectory(b'dir-id', 'dir2', root_id)), ('dir/child', 'other/child', b'child-id', inventory.InventoryFile(b'child-id', 'child', b'other-id'))])
        self.assertEqual('dir2', wt.id2path(b'dir-id'))
        self.assertEqual('other/child', wt.id2path(b'child-id'))

    def test_replace_root(self):
        wt = self.make_branch_and_tree('.')
        wt.lock_write()
        self.addCleanup(wt.unlock)
        root_id = wt.path2id('')
        wt.apply_inventory_delta([('', None, root_id, None), (None, '', b'root-id', inventory.InventoryDirectory(b'root-id', '', None))])
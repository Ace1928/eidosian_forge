import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_unshelve_subdir_in_now_removed_dir(self):
    tree = self.make_branch_and_tree('.')
    self.addCleanup(tree.lock_write().unlock)
    self.build_tree(['dir/', 'dir/subdir/', 'dir/subdir/foo'])
    tree.add(['dir'], ids=[b'dir-id'])
    tree.commit('versioned dir')
    tree.add(['dir/subdir', 'dir/subdir/foo'], ids=[b'subdir-id', b'foo-id'])
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    for change in creator.iter_shelvable():
        creator.shelve_change(change)
    shelf_manager = tree.get_shelf_manager()
    shelf_id = shelf_manager.shelve_changes(creator)
    self.assertPathDoesNotExist('dir/subdir')
    tree.remove(['dir'])
    unshelver = shelf_manager.get_unshelver(shelf_id)
    self.addCleanup(unshelver.finalize)
    unshelver.make_merger().do_merge()
    self.assertPathExists('dir/subdir/foo')
    self.assertEqual(b'dir-id', tree.path2id('dir'))
    self.assertEqual(b'subdir-id', tree.path2id('dir/subdir'))
    self.assertEqual(b'foo-id', tree.path2id('dir/subdir/foo'))
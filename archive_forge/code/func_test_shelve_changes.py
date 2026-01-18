import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_changes(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('no-change commit')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('tree/foo', b'bar')])
    self.assertFileEqual(b'bar', 'tree/foo')
    tree.add('foo', ids=b'foo-id')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    list(creator.iter_shelvable())
    creator.shelve_creation(b'foo-id')
    shelf_manager = tree.get_shelf_manager()
    shelf_id = shelf_manager.shelve_changes(creator)
    self.assertPathDoesNotExist('tree/foo')
    unshelver = shelf_manager.get_unshelver(shelf_id)
    self.addCleanup(unshelver.finalize)
    unshelver.make_merger().do_merge()
    self.assertFileEqual(b'bar', 'tree/foo')
import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_unshelve_base(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.commit('rev1', rev_id=b'rev1')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    manager = tree.get_shelf_manager()
    shelf_id, shelf_file = manager.new_shelf()
    try:
        creator.write_shelf(shelf_file)
    finally:
        shelf_file.close()
    tree.commit('rev2', rev_id=b'rev2')
    shelf_file = manager.read_shelf(1)
    self.addCleanup(shelf_file.close)
    unshelver = shelf.Unshelver.from_tree_and_shelf(tree, shelf_file)
    self.addCleanup(unshelver.finalize)
    self.assertEqual(b'rev1', unshelver.base_tree.get_revision_id())
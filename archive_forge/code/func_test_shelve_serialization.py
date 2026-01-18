import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_serialization(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    shelf_file = open('shelf', 'wb')
    self.addCleanup(shelf_file.close)
    try:
        creator.write_shelf(shelf_file)
    finally:
        shelf_file.close()
    self.assertFileEqual(EMPTY_SHELF, 'shelf')
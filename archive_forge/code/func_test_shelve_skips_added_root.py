import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_skips_added_root(self):
    """Skip adds of the root when iterating through shelvable changes."""
    tree = self.make_branch_and_tree('tree')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([], list(creator.iter_shelvable()))
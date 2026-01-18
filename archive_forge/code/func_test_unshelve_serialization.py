import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_unshelve_serialization(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('shelf', EMPTY_SHELF)])
    shelf_file = open('shelf', 'rb')
    self.addCleanup(shelf_file.close)
    unshelver = shelf.Unshelver.from_tree_and_shelf(tree, shelf_file)
    unshelver.finalize()
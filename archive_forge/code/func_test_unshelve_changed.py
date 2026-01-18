import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_unshelve_changed(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('tree/foo', b'a\nb\nc\n')])
    tree.add('foo', ids=b'foo-id')
    tree.commit('first commit')
    self.build_tree_contents([('tree/foo', b'a\nb\nd\n')])
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    list(creator.iter_shelvable())
    creator.shelve_lines(b'foo-id', [b'a\n', b'b\n', b'c\n'])
    shelf_file = open('shelf', 'w+b')
    self.addCleanup(shelf_file.close)
    creator.write_shelf(shelf_file)
    creator.transform()
    self.build_tree_contents([('tree/foo', b'z\na\nb\nc\n')])
    shelf_file.seek(0)
    unshelver = shelf.Unshelver.from_tree_and_shelf(tree, shelf_file)
    self.addCleanup(unshelver.finalize)
    unshelver.make_merger().do_merge()
    self.assertFileEqual(b'z\na\nb\nd\n', 'tree/foo')
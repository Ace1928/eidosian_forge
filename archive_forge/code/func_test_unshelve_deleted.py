import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_unshelve_deleted(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('tree/foo/',), ('tree/foo/bar', b'baz')])
    tree.add(['foo', 'foo/bar'], ids=[b'foo-id', b'bar-id'])
    tree.commit('Added file and directory')
    tree.unversion(['foo', 'foo/bar'])
    os.unlink('tree/foo/bar')
    os.rmdir('tree/foo')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    list(creator.iter_shelvable())
    creator.shelve_deletion(b'foo-id')
    creator.shelve_deletion(b'bar-id')
    with open('shelf', 'w+b') as shelf_file:
        creator.write_shelf(shelf_file)
        creator.transform()
        creator.finalize()
    self.assertEqual(tree.id2path(b'foo-id'), 'foo')
    self.assertEqual(tree.id2path(b'bar-id'), 'foo/bar')
    self.assertFileEqual(b'baz', 'tree/foo/bar')
    with open('shelf', 'r+b') as shelf_file:
        unshelver = shelf.Unshelver.from_tree_and_shelf(tree, shelf_file)
        self.addCleanup(unshelver.finalize)
        unshelver.make_merger().do_merge()
    self.assertRaises(errors.NoSuchId, tree.id2path, b'foo-id')
    self.assertRaises(errors.NoSuchId, tree.id2path, b'bar-id')
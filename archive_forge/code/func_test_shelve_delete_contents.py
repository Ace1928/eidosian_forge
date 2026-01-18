import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_delete_contents(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/foo'])
    tree.add('foo', ids=b'foo-id')
    tree.commit('Added file and directory')
    os.unlink('tree/foo')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('delete file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
    creator.shelve_deletion(b'foo-id')
    creator.transform()
    self.assertPathExists('tree/foo')
import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_creation_no_contents(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.commit('Empty tree')
    self.build_tree(['foo'])
    tree.add('foo', ids=b'foo-id')
    os.unlink('foo')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('add file', b'foo-id', None, 'foo')], sorted(list(creator.iter_shelvable())))
    creator.shelve_creation(b'foo-id')
    creator.transform()
    self.assertRaises(StopIteration, next, tree.iter_entries_by_dir(specific_files=['foo']))
    self.assertShelvedFileEqual('', creator, b'foo-id')
    s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    self.assertEqual(b'foo-id', creator.shelf_transform.final_file_id(s_trans_id))
    self.assertPathDoesNotExist('foo')
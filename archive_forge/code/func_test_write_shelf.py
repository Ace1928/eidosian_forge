import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_write_shelf(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/foo'])
    tree.add('foo', ids=b'foo-id')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    list(creator.iter_shelvable())
    creator.shelve_creation(b'foo-id')
    with open('shelf', 'wb') as shelf_file:
        creator.write_shelf(shelf_file)
    parser = pack.ContainerPushParser()
    with open('shelf', 'rb') as shelf_file:
        parser.accept_bytes(shelf_file.read())
    tt = tree.preview_transform()
    self.addCleanup(tt.finalize)
    records = iter(parser.read_pending_records())
    next(records)
    tt.deserialize(records)
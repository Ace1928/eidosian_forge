import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_creation(self):
    creator, tree = self.prepare_shelve_creation()
    creator.shelve_change(('add file', b'foo-id', 'file', 'foo'))
    creator.shelve_change(('add file', b'bar-id', 'directory', 'bar'))
    creator.transform()
    self.check_shelve_creation(creator, tree)
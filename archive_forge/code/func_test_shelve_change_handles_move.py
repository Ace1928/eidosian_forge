import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_move(self):
    creator, tree = self.prepare_shelve_move()
    creator.shelve_change(('rename', b'baz-id', 'foo/baz', 'bar/baz'))
    self.check_shelve_move(creator, tree)
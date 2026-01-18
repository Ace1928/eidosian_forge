import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_rename(self):
    creator = self.prepare_shelve_rename()
    creator.shelve_change(('rename', b'foo-id', 'foo', 'bar'))
    self.check_shelve_rename(creator)
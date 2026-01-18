import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_change_kind(self):
    creator = self.prepare_shelve_change_kind()
    creator.shelve_change(('change kind', b'foo-id', 'file', 'directory', 'foo'))
    creator.transform()
    self.check_shelve_change_kind(creator)
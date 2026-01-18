import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_shelve_change_handles_modify_text(self):
    creator = self.prepare_content_change()
    creator.shelve_change(('modify text', b'foo-id'))
    creator.transform()
    self.assertFileEqual(b'a\n', 'foo')
    self.assertShelvedFileEqual(b'b\na\nc\n', creator, b'foo-id')
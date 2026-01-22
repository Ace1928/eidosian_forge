import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def assertShelvedFileEqual(self, expected_content, creator, file_id):
    s_trans_id = creator.shelf_transform.trans_id_file_id(file_id)
    shelf_file = creator.shelf_transform._limbo_name(s_trans_id)
    self.assertFileEqual(expected_content, shelf_file)
import stat
from base64 import standard_b64encode
from dulwich.objects import Blob, Tree
from dulwich.repo import MemoryRepo as GitMemoryRepo
from ...revision import Revision
from ...tests import TestCase
from ..pristine_tar import (get_pristine_tar_tree, read_git_pristine_tar_data,
def test_read_pristine_tar_data_no_branch(self):
    r = GitMemoryRepo()
    self.assertRaises(KeyError, read_git_pristine_tar_data, r, b'foo')
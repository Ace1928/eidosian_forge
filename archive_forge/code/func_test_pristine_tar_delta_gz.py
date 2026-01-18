import stat
from base64 import standard_b64encode
from dulwich.objects import Blob, Tree
from dulwich.repo import MemoryRepo as GitMemoryRepo
from ...revision import Revision
from ...tests import TestCase
from ..pristine_tar import (get_pristine_tar_tree, read_git_pristine_tar_data,
def test_pristine_tar_delta_gz(self):
    rev = Revision(b'myrevid')
    rev.properties['deb-pristine-delta'] = standard_b64encode(b'bla')
    self.assertEqual((b'bla', 'gz'), revision_pristine_tar_data(rev))
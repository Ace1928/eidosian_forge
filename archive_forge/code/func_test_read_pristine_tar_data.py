import stat
from base64 import standard_b64encode
from dulwich.objects import Blob, Tree
from dulwich.repo import MemoryRepo as GitMemoryRepo
from ...revision import Revision
from ...tests import TestCase
from ..pristine_tar import (get_pristine_tar_tree, read_git_pristine_tar_data,
def test_read_pristine_tar_data(self):
    r = GitMemoryRepo()
    delta = Blob.from_string(b'some yummy data')
    r.object_store.add_object(delta)
    idfile = Blob.from_string(b'someid')
    r.object_store.add_object(idfile)
    t = Tree()
    t.add(b'foo.delta', stat.S_IFREG | 420, delta.id)
    t.add(b'foo.id', stat.S_IFREG | 420, idfile.id)
    r.object_store.add_object(t)
    r.do_commit(b'pristine tar delta for foo', tree=t.id, ref=b'refs/heads/pristine-tar')
    self.assertEqual((b'some yummy data', b'someid'), read_git_pristine_tar_data(r, b'foo'))
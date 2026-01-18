from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_remove_bin_blob(self):
    f = BytesIO()
    b1 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b')
    store = MemoryObjectStore()
    store.add_object(b1)
    write_object_diff(f, store, (b'foo.png', 420, b1.id), (None, None, None))
    self.assertEqual([b'diff --git a/foo.png b/foo.png', b'deleted file mode 644', b'index f73e47d..0000000', b'Binary files a/foo.png and /dev/null differ'], f.getvalue().splitlines())
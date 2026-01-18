from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_blob(self):
    f = BytesIO()
    b1 = Blob.from_string(b'old\nsame\n')
    b2 = Blob.from_string(b'new\nsame\n')
    store = MemoryObjectStore()
    store.add_objects([(b1, None), (b2, None)])
    write_object_diff(f, store, (b'foo.txt', 420, b1.id), (b'bar.txt', 420, b2.id))
    self.assertEqual([b'diff --git a/foo.txt b/bar.txt', b'index 3b0f961..a116b51 644', b'--- a/foo.txt', b'+++ b/bar.txt', b'@@ -1,2 +1,2 @@', b'-old', b'+new', b' same'], f.getvalue().splitlines())
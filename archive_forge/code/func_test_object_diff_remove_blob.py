from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
def test_object_diff_remove_blob(self):
    f = BytesIO()
    b1 = Blob.from_string(b'new\nsame\n')
    store = MemoryObjectStore()
    store.add_object(b1)
    write_object_diff(f, store, (b'bar.txt', 420, b1.id), (None, None, None))
    self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'deleted file mode 644', b'index a116b51..0000000', b'--- a/bar.txt', b'+++ /dev/null', b'@@ -1,2 +0,0 @@', b'-new', b'-same'], f.getvalue().splitlines())
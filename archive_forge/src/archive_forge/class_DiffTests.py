from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
class DiffTests(TestCase):
    """Tests for write_blob_diff and write_tree_diff."""

    def test_blob_diff(self):
        f = BytesIO()
        write_blob_diff(f, (b'foo.txt', 420, Blob.from_string(b'old\nsame\n')), (b'bar.txt', 420, Blob.from_string(b'new\nsame\n')))
        self.assertEqual([b'diff --git a/foo.txt b/bar.txt', b'index 3b0f961..a116b51 644', b'--- a/foo.txt', b'+++ b/bar.txt', b'@@ -1,2 +1,2 @@', b'-old', b'+new', b' same'], f.getvalue().splitlines())

    def test_blob_add(self):
        f = BytesIO()
        write_blob_diff(f, (None, None, None), (b'bar.txt', 420, Blob.from_string(b'new\nsame\n')))
        self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'new file mode 644', b'index 0000000..a116b51', b'--- /dev/null', b'+++ b/bar.txt', b'@@ -0,0 +1,2 @@', b'+new', b'+same'], f.getvalue().splitlines())

    def test_blob_remove(self):
        f = BytesIO()
        write_blob_diff(f, (b'bar.txt', 420, Blob.from_string(b'new\nsame\n')), (None, None, None))
        self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'deleted file mode 644', b'index a116b51..0000000', b'--- a/bar.txt', b'+++ /dev/null', b'@@ -1,2 +0,0 @@', b'-new', b'-same'], f.getvalue().splitlines())

    def test_tree_diff(self):
        f = BytesIO()
        store = MemoryObjectStore()
        added = Blob.from_string(b'add\n')
        removed = Blob.from_string(b'removed\n')
        changed1 = Blob.from_string(b'unchanged\nremoved\n')
        changed2 = Blob.from_string(b'unchanged\nadded\n')
        unchanged = Blob.from_string(b'unchanged\n')
        tree1 = Tree()
        tree1.add(b'removed.txt', 420, removed.id)
        tree1.add(b'changed.txt', 420, changed1.id)
        tree1.add(b'unchanged.txt', 420, changed1.id)
        tree2 = Tree()
        tree2.add(b'added.txt', 420, added.id)
        tree2.add(b'changed.txt', 420, changed2.id)
        tree2.add(b'unchanged.txt', 420, changed1.id)
        store.add_objects([(o, None) for o in [tree1, tree2, added, removed, changed1, changed2, unchanged]])
        write_tree_diff(f, store, tree1.id, tree2.id)
        self.assertEqual([b'diff --git a/added.txt b/added.txt', b'new file mode 644', b'index 0000000..76d4bb8', b'--- /dev/null', b'+++ b/added.txt', b'@@ -0,0 +1 @@', b'+add', b'diff --git a/changed.txt b/changed.txt', b'index bf84e48..1be2436 644', b'--- a/changed.txt', b'+++ b/changed.txt', b'@@ -1,2 +1,2 @@', b' unchanged', b'-removed', b'+added', b'diff --git a/removed.txt b/removed.txt', b'deleted file mode 644', b'index 2c3f0b3..0000000', b'--- a/removed.txt', b'+++ /dev/null', b'@@ -1 +0,0 @@', b'-removed'], f.getvalue().splitlines())

    def test_tree_diff_submodule(self):
        f = BytesIO()
        store = MemoryObjectStore()
        tree1 = Tree()
        tree1.add(b'asubmodule', S_IFGITLINK, b'06d0bdd9e2e20377b3180e4986b14c8549b393e4')
        tree2 = Tree()
        tree2.add(b'asubmodule', S_IFGITLINK, b'cc975646af69f279396d4d5e1379ac6af80ee637')
        store.add_objects([(o, None) for o in [tree1, tree2]])
        write_tree_diff(f, store, tree1.id, tree2.id)
        self.assertEqual([b'diff --git a/asubmodule b/asubmodule', b'index 06d0bdd..cc97564 160000', b'--- a/asubmodule', b'+++ b/asubmodule', b'@@ -1 +1 @@', b'-Subproject commit 06d0bdd9e2e20377b3180e4986b14c8549b393e4', b'+Subproject commit cc975646af69f279396d4d5e1379ac6af80ee637'], f.getvalue().splitlines())

    def test_object_diff_blob(self):
        f = BytesIO()
        b1 = Blob.from_string(b'old\nsame\n')
        b2 = Blob.from_string(b'new\nsame\n')
        store = MemoryObjectStore()
        store.add_objects([(b1, None), (b2, None)])
        write_object_diff(f, store, (b'foo.txt', 420, b1.id), (b'bar.txt', 420, b2.id))
        self.assertEqual([b'diff --git a/foo.txt b/bar.txt', b'index 3b0f961..a116b51 644', b'--- a/foo.txt', b'+++ b/bar.txt', b'@@ -1,2 +1,2 @@', b'-old', b'+new', b' same'], f.getvalue().splitlines())

    def test_object_diff_add_blob(self):
        f = BytesIO()
        store = MemoryObjectStore()
        b2 = Blob.from_string(b'new\nsame\n')
        store.add_object(b2)
        write_object_diff(f, store, (None, None, None), (b'bar.txt', 420, b2.id))
        self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'new file mode 644', b'index 0000000..a116b51', b'--- /dev/null', b'+++ b/bar.txt', b'@@ -0,0 +1,2 @@', b'+new', b'+same'], f.getvalue().splitlines())

    def test_object_diff_remove_blob(self):
        f = BytesIO()
        b1 = Blob.from_string(b'new\nsame\n')
        store = MemoryObjectStore()
        store.add_object(b1)
        write_object_diff(f, store, (b'bar.txt', 420, b1.id), (None, None, None))
        self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'deleted file mode 644', b'index a116b51..0000000', b'--- a/bar.txt', b'+++ /dev/null', b'@@ -1,2 +0,0 @@', b'-new', b'-same'], f.getvalue().splitlines())

    def test_object_diff_bin_blob_force(self):
        f = BytesIO()
        b1 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b')
        b2 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3')
        store = MemoryObjectStore()
        store.add_objects([(b1, None), (b2, None)])
        write_object_diff(f, store, (b'foo.png', 420, b1.id), (b'bar.png', 420, b2.id), diff_binary=True)
        self.assertEqual([b'diff --git a/foo.png b/bar.png', b'index f73e47d..06364b7 644', b'--- a/foo.png', b'+++ b/bar.png', b'@@ -1,4 +1,4 @@', b' \x89PNG', b' \x1a', b' \x00\x00\x00', b'-IHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b', b'\\ No newline at end of file', b'+IHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3', b'\\ No newline at end of file'], f.getvalue().splitlines())

    def test_object_diff_bin_blob(self):
        f = BytesIO()
        b1 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b')
        b2 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3')
        store = MemoryObjectStore()
        store.add_objects([(b1, None), (b2, None)])
        write_object_diff(f, store, (b'foo.png', 420, b1.id), (b'bar.png', 420, b2.id))
        self.assertEqual([b'diff --git a/foo.png b/bar.png', b'index f73e47d..06364b7 644', b'Binary files a/foo.png and b/bar.png differ'], f.getvalue().splitlines())

    def test_object_diff_add_bin_blob(self):
        f = BytesIO()
        b2 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x03\x00\x00\x00\x98\xd3\xb3')
        store = MemoryObjectStore()
        store.add_object(b2)
        write_object_diff(f, store, (None, None, None), (b'bar.png', 420, b2.id))
        self.assertEqual([b'diff --git a/bar.png b/bar.png', b'new file mode 644', b'index 0000000..06364b7', b'Binary files /dev/null and b/bar.png differ'], f.getvalue().splitlines())

    def test_object_diff_remove_bin_blob(self):
        f = BytesIO()
        b1 = Blob.from_string(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\xd5\x00\x00\x00\x9f\x08\x04\x00\x00\x00\x05\x04\x8b')
        store = MemoryObjectStore()
        store.add_object(b1)
        write_object_diff(f, store, (b'foo.png', 420, b1.id), (None, None, None))
        self.assertEqual([b'diff --git a/foo.png b/foo.png', b'deleted file mode 644', b'index f73e47d..0000000', b'Binary files a/foo.png and /dev/null differ'], f.getvalue().splitlines())

    def test_object_diff_kind_change(self):
        f = BytesIO()
        b1 = Blob.from_string(b'new\nsame\n')
        store = MemoryObjectStore()
        store.add_object(b1)
        write_object_diff(f, store, (b'bar.txt', 420, b1.id), (b'bar.txt', 57344, b'06d0bdd9e2e20377b3180e4986b14c8549b393e4'))
        self.assertEqual([b'diff --git a/bar.txt b/bar.txt', b'old file mode 644', b'new file mode 160000', b'index a116b51..06d0bdd 160000', b'--- a/bar.txt', b'+++ b/bar.txt', b'@@ -1,2 +1 @@', b'-new', b'-same', b'+Subproject commit 06d0bdd9e2e20377b3180e4986b14c8549b393e4'], f.getvalue().splitlines())
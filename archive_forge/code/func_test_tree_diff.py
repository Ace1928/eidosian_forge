from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
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
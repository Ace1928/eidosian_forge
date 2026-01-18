from io import BytesIO, StringIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..patch import (
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
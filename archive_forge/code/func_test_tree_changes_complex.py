from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_complex(self):
    blob_a_1 = make_object(Blob, data=b'a1_1')
    blob_bx1_1 = make_object(Blob, data=b'bx1_1')
    blob_bx2_1 = make_object(Blob, data=b'bx2_1')
    blob_by1_1 = make_object(Blob, data=b'by1_1')
    blob_by2_1 = make_object(Blob, data=b'by2_1')
    tree1 = self.commit_tree([(b'a', blob_a_1), (b'b/x/1', blob_bx1_1), (b'b/x/2', blob_bx2_1), (b'b/y/1', blob_by1_1), (b'b/y/2', blob_by2_1)])
    blob_a_2 = make_object(Blob, data=b'a1_2')
    blob_bx1_2 = blob_bx1_1
    blob_by_2 = make_object(Blob, data=b'by_2')
    blob_c_2 = make_object(Blob, data=b'c_2')
    tree2 = self.commit_tree([(b'a', blob_a_2), (b'b/x/1', blob_bx1_2), (b'b/y', blob_by_2), (b'c', blob_c_2)])
    self.assertChangesEqual([TreeChange(CHANGE_MODIFY, (b'a', F, blob_a_1.id), (b'a', F, blob_a_2.id)), TreeChange.delete((b'b/x/2', F, blob_bx2_1.id)), TreeChange.add((b'b/y', F, blob_by_2.id)), TreeChange.delete((b'b/y/1', F, blob_by1_1.id)), TreeChange.delete((b'b/y/2', F, blob_by2_1.id)), TreeChange.add((b'c', F, blob_c_2.id))], tree1, tree2)
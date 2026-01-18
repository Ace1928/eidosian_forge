from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_changes_name_order(self):
    blob = make_object(Blob, data=b'a')
    tree1 = self.commit_tree([(b'a', blob), (b'a.', blob), (b'a..', blob)])
    tree2 = self.commit_tree([(b'a/x', blob), (b'a./x', blob), (b'a..', blob)])
    self.assertChangesEqual([TreeChange.delete((b'a', F, blob.id)), TreeChange.add((b'a/x', F, blob.id)), TreeChange.delete((b'a.', F, blob.id)), TreeChange.add((b'a./x', F, blob.id))], tree1, tree2)
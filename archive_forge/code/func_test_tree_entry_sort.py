from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_tree_entry_sort(self):
    sha = 'abcd' * 10
    expected_entries = [TreeChange.add(TreeEntry(b'aaa', F, sha)), TreeChange(CHANGE_COPY, TreeEntry(b'bbb', F, sha), TreeEntry(b'aab', F, sha)), TreeChange(CHANGE_MODIFY, TreeEntry(b'bbb', F, sha), TreeEntry(b'bbb', F, b'dabc' * 10)), TreeChange(CHANGE_RENAME, TreeEntry(b'bbc', F, sha), TreeEntry(b'ddd', F, sha)), TreeChange.delete(TreeEntry(b'ccc', F, sha))]
    for perm in permutations(expected_entries):
        self.assertEqual(expected_entries, sorted(perm, key=_tree_change_key))
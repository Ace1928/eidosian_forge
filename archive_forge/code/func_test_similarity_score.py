from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def test_similarity_score(self):
    blob0 = make_object(Blob, data=b'')
    blob1 = make_object(Blob, data=b'ab\ncd\ncd\n')
    blob2 = make_object(Blob, data=b'ab\n')
    blob3 = make_object(Blob, data=b'cd\n')
    blob4 = make_object(Blob, data=b'cd\ncd\n')
    self.assertSimilar(100, blob0, blob0)
    self.assertSimilar(0, blob0, blob1)
    self.assertSimilar(33, blob1, blob2)
    self.assertSimilar(33, blob1, blob3)
    self.assertSimilar(66, blob1, blob4)
    self.assertSimilar(0, blob2, blob3)
    self.assertSimilar(50, blob3, blob4)
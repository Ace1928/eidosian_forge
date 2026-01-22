from itertools import permutations
from dulwich.tests import TestCase
from ..diff_tree import (
from ..index import commit_tree
from ..object_store import MemoryObjectStore
from ..objects import Blob, ShaFile, Tree, TreeEntry
from .utils import F, ext_functest_builder, functest_builder, make_object
def _do_test_count_blocks_chunks(self, count_blocks):
    blob = ShaFile.from_raw_chunks(Blob.type_num, [b'a\nb', b'\na\n'])
    self.assertBlockCountEqual({b'a\n': 4, b'b\n': 2}, _count_blocks(blob))
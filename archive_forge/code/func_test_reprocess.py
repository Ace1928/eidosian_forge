from breezy.tests import TestCase
from breezy.textmerge import Merge2
def test_reprocess(self):
    struct = [('a', 'b'), ('c',), ('def', 'geh'), ('i',)]
    expect = [('a', 'b'), ('c',), ('d', 'g'), ('e',), ('f', 'h'), ('i',)]
    result = Merge2.reprocess_struct(struct)
    self.assertEqual(list(result), expect)
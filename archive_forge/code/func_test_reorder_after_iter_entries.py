from ... import errors, tests, transport
from .. import index as _mod_index
def test_reorder_after_iter_entries(self):
    idx = _mod_index.CombinedGraphIndex([])
    idx.insert_index(0, self.make_index_with_simple_nodes('1'), b'1')
    idx.insert_index(1, self.make_index_with_simple_nodes('2'), b'2')
    idx.insert_index(2, self.make_index_with_simple_nodes('3'), b'3')
    idx.insert_index(3, self.make_index_with_simple_nodes('4'), b'4')
    idx1, idx2, idx3, idx4 = idx._indices
    self.assertLength(2, list(idx.iter_entries([(b'index-4-key-1',), (b'index-2-key-1',)])))
    self.assertEqual([idx2, idx4, idx1, idx3], idx._indices)
    self.assertEqual([b'2', b'4', b'1', b'3'], idx._index_names)
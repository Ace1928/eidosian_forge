from ... import errors, tests, transport
from .. import index as _mod_index
def test_reorder_propagates_to_siblings(self):
    cgi1 = _mod_index.CombinedGraphIndex([])
    cgi2 = _mod_index.CombinedGraphIndex([])
    cgi1.insert_index(0, self.make_index_with_simple_nodes('1-1'), 'one')
    cgi1.insert_index(1, self.make_index_with_simple_nodes('1-2'), 'two')
    cgi2.insert_index(0, self.make_index_with_simple_nodes('2-1'), 'one')
    cgi2.insert_index(1, self.make_index_with_simple_nodes('2-2'), 'two')
    index2_1, index2_2 = cgi2._indices
    cgi1.set_sibling_indices([cgi2])
    list(cgi1.iter_entries([(b'index-1-2-key-1',)]))
    self.assertEqual([index2_2, index2_1], cgi2._indices)
    self.assertEqual(['two', 'one'], cgi2._index_names)
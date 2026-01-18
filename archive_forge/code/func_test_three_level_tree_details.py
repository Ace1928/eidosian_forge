import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_three_level_tree_details(self):
    self.shrink_page_size()
    builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
    nodes = self.make_nodes(20000, 2, 2)
    for node in nodes:
        builder.add_node(*node)
    t = transport.get_transport_from_url('trace+' + self.get_url(''))
    size = t.put_file('index', self.time(builder.finish))
    del builder
    index = btree_index.BTreeGraphIndex(t, 'index', size)
    index.key_count()
    self.assertEqual(3, len(index._row_lengths), 'Not enough rows: %r' % index._row_lengths)
    self.assertEqual(4, len(index._row_offsets))
    self.assertEqual(sum(index._row_lengths), index._row_offsets[-1])
    internal_nodes = index._get_internal_nodes([0, 1, 2])
    root_node = internal_nodes[0]
    internal_node1 = internal_nodes[1]
    internal_node2 = internal_nodes[2]
    self.assertEqual(internal_node2.offset, 1 + len(internal_node1.keys))
    pos = index._row_offsets[2] + internal_node2.offset + 1
    leaf = index._get_leaf_nodes([pos])[pos]
    self.assertTrue(internal_node2.keys[0] in leaf)
import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def test_root_leaf_1_0(self):
    builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
    nodes = self.make_nodes(5, 1, 0)
    for node in nodes:
        builder.add_node(*node)
    temp_file = builder.finish()
    content = temp_file.read()
    del temp_file
    self.assertEqual(131, len(content))
    self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=5\nrow_lengths=1\n', content[:73])
    node_content = content[73:]
    node_bytes = zlib.decompress(node_content)
    expected_node = b'type=leaf\n0000000000000000000000000000000000000000\x00\x00value:0\n1111111111111111111111111111111111111111\x00\x00value:1\n2222222222222222222222222222222222222222\x00\x00value:2\n3333333333333333333333333333333333333333\x00\x00value:3\n4444444444444444444444444444444444444444\x00\x00value:4\n'
    self.assertEqual(expected_node, node_bytes)
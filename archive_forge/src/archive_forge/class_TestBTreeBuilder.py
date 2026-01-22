import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class TestBTreeBuilder(BTreeTestCase):

    def test_clear_cache(self):
        builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=1)
        builder.clear_cache()

    def test_empty_1_0(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=0\nrow_lengths=\n', content)

    def test_empty_2_1(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=1)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=1\nkey_elements=2\nlen=0\nrow_lengths=\n', content)

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

    def test_root_leaf_2_2(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(5, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqual(238, len(content))
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=2\nkey_elements=2\nlen=10\nrow_lengths=1\n', content[:74])
        node_content = content[74:]
        node_bytes = zlib.decompress(node_content)
        expected_node = b'type=leaf\n0000000000000000000000000000000000000000\x000000000000000000000000000000000000000000\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:0\n0000000000000000000000000000000000000000\x001111111111111111111111111111111111111111\x000000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\r0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:1\n0000000000000000000000000000000000000000\x002222222222222222222222222222222222222222\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:2\n0000000000000000000000000000000000000000\x003333333333333333333333333333333333333333\x000000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\t0000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\r0000000000000000000000000000000000000000\x00ref2222222222222222222222222222222222222222\x00value:3\n0000000000000000000000000000000000000000\x004444444444444444444444444444444444444444\x00\t0000000000000000000000000000000000000000\x00ref0000000000000000000000000000000000000000\x00value:4\n1111111111111111111111111111111111111111\x000000000000000000000000000000000000000000\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:0\n1111111111111111111111111111111111111111\x001111111111111111111111111111111111111111\x001111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\r1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:1\n1111111111111111111111111111111111111111\x002222222222222222222222222222222222222222\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:2\n1111111111111111111111111111111111111111\x003333333333333333333333333333333333333333\x001111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\t1111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\r1111111111111111111111111111111111111111\x00ref2222222222222222222222222222222222222222\x00value:3\n1111111111111111111111111111111111111111\x004444444444444444444444444444444444444444\x00\t1111111111111111111111111111111111111111\x00ref0000000000000000000000000000000000000000\x00value:4\n'
        self.assertEqual(expected_node, node_bytes)

    def test_2_leaves_1_0(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(400, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(9283, len(content))
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=400\nrow_lengths=1,2\n', content[:77])
        root = content[77:4096]
        leaf1 = content[4096:8192]
        leaf2 = content[8192:]
        root_bytes = zlib.decompress(root)
        expected_root = b'type=internal\noffset=0\n' + b'307' * 40 + b'\n'
        self.assertEqual(expected_root, root_bytes)
        leaf1_bytes = zlib.decompress(leaf1)
        sorted_node_keys = sorted((node[0] for node in nodes))
        node = btree_index._LeafNode(leaf1_bytes, 1, 0)
        self.assertEqual(231, len(node))
        self.assertEqual(sorted_node_keys[:231], node.all_keys())
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(400 - 231, len(node))
        self.assertEqual(sorted_node_keys[231:], node.all_keys())

    def test_last_page_rounded_1_layer(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(10, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(155, len(content))
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=10\nrow_lengths=1\n', content[:74])
        leaf2 = content[74:]
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(10, len(node))
        sorted_node_keys = sorted((node[0] for node in nodes))
        self.assertEqual(sorted_node_keys, node.all_keys())

    def test_last_page_not_rounded_2_layer(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        nodes = self.make_nodes(400, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(9283, len(content))
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=0\nkey_elements=1\nlen=400\nrow_lengths=1,2\n', content[:77])
        leaf2 = content[8192:]
        leaf2_bytes = zlib.decompress(leaf2)
        node = btree_index._LeafNode(leaf2_bytes, 1, 0)
        self.assertEqual(400 - 231, len(node))
        sorted_node_keys = sorted((node[0] for node in nodes))
        self.assertEqual(sorted_node_keys[231:], node.all_keys())

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

    def test_2_leaves_2_2(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(100, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        self.assertEqualApproxCompressed(12643, len(content))
        self.assertEqual(b'B+Tree Graph Index 2\nnode_ref_lists=2\nkey_elements=2\nlen=200\nrow_lengths=1,3\n', content[:77])
        root = content[77:4096]
        leaf1 = content[4096:8192]
        leaf2 = content[8192:12288]
        leaf3 = content[12288:]
        root_bytes = zlib.decompress(root)
        expected_root = b'type=internal\noffset=0\n' + b'0' * 40 + b'\x00' + b'91' * 40 + b'\n' + b'1' * 40 + b'\x00' + b'81' * 40 + b'\n'
        self.assertEqual(expected_root, root_bytes)

    def test_spill_index_stress_1_1(self):
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[12])
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(4, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(None, builder._backing_indices[2])
        self.assertEqual(16, builder._backing_indices[3].key_count())
        t = self.get_transport('')
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_spill_index_stress_1_1_no_combine(self):
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        builder.set_optimize(for_size=False, combine_backing_indices=False)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(3, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        builder.add_node(*nodes[12])
        self.assertEqual(6, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(8, len(builder._backing_indices))
        for backing_index in builder._backing_indices:
            self.assertEqual(2, backing_index.key_count())
        transport = self.get_transport('')
        size = transport.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(transport, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_set_optimize(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        builder.set_optimize(for_size=True)
        self.assertTrue(builder._optimize_for_size)
        builder.set_optimize(for_size=False)
        self.assertFalse(builder._optimize_for_size)
        obj = object()
        builder._optimize_for_size = obj
        builder.set_optimize(combine_backing_indices=False)
        self.assertFalse(builder._combine_backing_indices)
        self.assertIs(obj, builder._optimize_for_size)
        builder.set_optimize(combine_backing_indices=True)
        self.assertTrue(builder._combine_backing_indices)
        self.assertIs(obj, builder._optimize_for_size)

    def test_spill_index_stress_2_2(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2, spill_at=2)
        nodes = self.make_nodes(16, 2, 2)
        builder.add_node(*nodes[0])
        self.assertEqual(1, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        builder.add_node(*nodes[1])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(1, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        old = dict(builder._get_nodes_by_key())
        builder.add_node(*nodes[2])
        self.assertEqual(1, len(builder._nodes))
        self.assertIsNot(None, builder._nodes_by_key)
        self.assertNotEqual({}, builder._nodes_by_key)
        self.assertNotEqual(old, builder._nodes_by_key)
        builder.add_node(*nodes[3])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[4])
        builder.add_node(*nodes[5])
        self.assertEqual(0, len(builder._nodes))
        self.assertIs(None, builder._nodes_by_key)
        self.assertEqual(2, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        builder.add_node(*nodes[6])
        builder.add_node(*nodes[7])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[8])
        builder.add_node(*nodes[9])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[10])
        builder.add_node(*nodes[11])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[12])
        self.assertEqual([(builder,) + node for node in sorted(nodes[:13])], list(builder.iter_all_entries()))
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries([nodes[12][0], nodes[11][0]])))
        self.assertEqual(13, builder.key_count())
        self.assertEqual({(builder,) + node for node in nodes[11:13]}, set(builder.iter_entries_prefix([nodes[12][0], nodes[11][0]])))
        builder.add_node(*nodes[13])
        self.assertEqual(3, len(builder._backing_indices))
        self.assertEqual(2, builder._backing_indices[0].key_count())
        self.assertEqual(4, builder._backing_indices[1].key_count())
        self.assertEqual(8, builder._backing_indices[2].key_count())
        builder.add_node(*nodes[14])
        builder.add_node(*nodes[15])
        self.assertEqual(4, len(builder._backing_indices))
        self.assertEqual(None, builder._backing_indices[0])
        self.assertEqual(None, builder._backing_indices[1])
        self.assertEqual(None, builder._backing_indices[2])
        self.assertEqual(16, builder._backing_indices[3].key_count())
        transport = self.get_transport('')
        size = transport.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(transport, 'index', size)
        nodes = list(index.iter_all_entries())
        self.assertEqual(sorted(nodes), nodes)
        self.assertEqual(16, len(nodes))

    def test_spill_index_duplicate_key_caught_on_finish(self):
        builder = btree_index.BTreeBuilder(key_elements=1, spill_at=2)
        nodes = [node[0:2] for node in self.make_nodes(16, 1, 0)]
        builder.add_node(*nodes[0])
        builder.add_node(*nodes[1])
        builder.add_node(*nodes[0])
        self.assertRaises(_mod_index.BadIndexDuplicateKey, builder.finish)
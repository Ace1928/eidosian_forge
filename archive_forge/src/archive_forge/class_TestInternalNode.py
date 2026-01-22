from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestInternalNode(TestCaseWithStore):

    def test_add_node_empty_new(self):
        node = InternalNode(b'fo')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'bar')
        node.add_node(b'foo', child)
        self.assertEqual(3, node._node_width)
        self.assertEqual({(b'foo',): b'bar'}, self.to_dict(node, None))
        self.assertEqual(1, len(node))
        chk_bytes = self.get_chk_bytes()
        keys = list(node.serialise(chk_bytes))
        child_key = child.serialise(chk_bytes)[0]
        self.assertEqual([child_key, (b'sha1:cf67e9997d8228a907c1f5bfb25a8bd9cd916fac',)], keys)
        bytes = self.read_bytes(chk_bytes, keys[1])
        node = chk_map._deserialise(bytes, keys[1], None)
        self.assertEqual(1, len(node))
        self.assertEqual({(b'foo',): b'bar'}, self.to_dict(node, chk_bytes))
        self.assertEqual(3, node._node_width)

    def test_add_node_resets_key_new(self):
        node = InternalNode(b'fo')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'bar')
        node.add_node(b'foo', child)
        chk_bytes = self.get_chk_bytes()
        keys = list(node.serialise(chk_bytes))
        self.assertEqual(keys[1], node._key)
        node.add_node(b'fos', child)
        self.assertEqual(None, node._key)

    def test__iter_nodes_no_key_filter(self):
        node = InternalNode(b'')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'bar')
        node.add_node(b'f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'bar',), b'baz')
        node.add_node(b'b', child)
        for child, node_key_filter in node._iter_nodes(None, key_filter=None):
            self.assertEqual(None, node_key_filter)

    def test__iter_nodes_splits_key_filter(self):
        node = InternalNode(b'')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'bar')
        node.add_node(b'f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'bar',), b'baz')
        node.add_node(b'b', child)
        key_filter = ((b'foo',), (b'bar',), (b'cat',))
        for child, node_key_filter in node._iter_nodes(None, key_filter=key_filter):
            self.assertEqual(1, len(node_key_filter))

    def test__iter_nodes_with_multiple_matches(self):
        node = InternalNode(b'')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'val')
        child.map(None, (b'fob',), b'val')
        node.add_node(b'f', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'bar',), b'val')
        child.map(None, (b'baz',), b'val')
        node.add_node(b'b', child)
        key_filter = ((b'foo',), (b'fob',), (b'bar',), (b'baz',), (b'ram',))
        for child, node_key_filter in node._iter_nodes(None, key_filter=key_filter):
            self.assertEqual(2, len(node_key_filter))

    def make_fo_fa_node(self):
        node = InternalNode(b'f')
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'foo',), b'val')
        child.map(None, (b'fob',), b'val')
        node.add_node(b'fo', child)
        child = LeafNode()
        child.set_maximum_size(100)
        child.map(None, (b'far',), b'val')
        child.map(None, (b'faz',), b'val')
        node.add_node(b'fa', child)
        return node

    def test__iter_nodes_single_entry(self):
        node = self.make_fo_fa_node()
        key_filter = [(b'foo',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(1, len(nodes))
        self.assertEqual(key_filter, nodes[0][1])

    def test__iter_nodes_single_entry_misses(self):
        node = self.make_fo_fa_node()
        key_filter = [(b'bar',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(0, len(nodes))

    def test__iter_nodes_mixed_key_width(self):
        node = self.make_fo_fa_node()
        key_filter = [(b'foo', b'bar'), (b'foo',), (b'fo',), (b'b',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(1, len(nodes))
        matches = key_filter[:]
        matches.remove((b'b',))
        self.assertEqual(sorted(matches), sorted(nodes[0][1]))

    def test__iter_nodes_match_all(self):
        node = self.make_fo_fa_node()
        key_filter = [(b'foo', b'bar'), (b'foo',), (b'fo',), (b'f',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(2, len(nodes))

    def test__iter_nodes_fixed_widths_and_misses(self):
        node = self.make_fo_fa_node()
        key_filter = [(b'foo',), (b'faa',), (b'baz',)]
        nodes = list(node._iter_nodes(None, key_filter=key_filter))
        self.assertEqual(2, len(nodes))
        for node, matches in nodes:
            self.assertEqual(1, len(matches))

    def test_iteritems_empty_new(self):
        node = InternalNode()
        self.assertEqual([], sorted(node.iteritems(None)))

    def test_iteritems_two_children(self):
        node = InternalNode()
        leaf1 = LeafNode()
        leaf1.map(None, (b'foo bar',), b'quux')
        leaf2 = LeafNode()
        leaf2.map(None, (b'strange',), b'beast')
        node.add_node(b'f', leaf1)
        node.add_node(b's', leaf2)
        self.assertEqual([((b'foo bar',), b'quux'), ((b'strange',), b'beast')], sorted(node.iteritems(None)))

    def test_iteritems_two_children_partial(self):
        node = InternalNode()
        leaf1 = LeafNode()
        leaf1.map(None, (b'foo bar',), b'quux')
        leaf2 = LeafNode()
        leaf2.map(None, (b'strange',), b'beast')
        node.add_node(b'f', leaf1)
        node._items[b'f'] = None
        node.add_node(b's', leaf2)
        self.assertEqual([((b'strange',), b'beast')], sorted(node.iteritems(None, [(b'strange',), (b'weird',)])))

    def test_iteritems_two_children_with_hash(self):
        search_key_func = chk_map.search_key_registry.get(b'hash-255-way')
        node = InternalNode(search_key_func=search_key_func)
        leaf1 = LeafNode(search_key_func=search_key_func)
        leaf1.map(None, StaticTuple(b'foo bar'), b'quux')
        leaf2 = LeafNode(search_key_func=search_key_func)
        leaf2.map(None, StaticTuple(b'strange'), b'beast')
        self.assertEqual(b'\xbeF\x014', search_key_func(StaticTuple(b'foo bar')))
        self.assertEqual(b'\x85\xfa\xf7K', search_key_func(StaticTuple(b'strange')))
        node.add_node(b'\xbe', leaf1)
        node._items[b'\xbe'] = None
        node.add_node(b'\x85', leaf2)
        self.assertEqual([((b'strange',), b'beast')], sorted(node.iteritems(None, [StaticTuple(b'strange'), StaticTuple(b'weird')])))

    def test_iteritems_partial_empty(self):
        node = InternalNode()
        self.assertEqual([], sorted(node.iteritems([(b'missing',)])))

    def test_map_to_new_child_new(self):
        chkmap = self._get_map({(b'k1',): b'foo', (b'k2',): b'bar'}, maximum_size=10)
        chkmap._ensure_root()
        node = chkmap._root_node
        self.assertEqual(2, len([value for value in node._items.values() if isinstance(value, StaticTuple)]))
        prefix, nodes = node.map(None, (b'k3',), b'quux')
        self.assertEqual(b'k', prefix)
        self.assertEqual([(b'', node)], nodes)
        child = node._items[b'k3']
        self.assertIsInstance(child, LeafNode)
        self.assertEqual(1, len(child))
        self.assertEqual({(b'k3',): b'quux'}, self.to_dict(child, None))
        self.assertEqual(None, child._key)
        self.assertEqual(10, child.maximum_size)
        self.assertEqual(1, child._key_width)
        self.assertEqual(3, len(chkmap))
        self.assertEqual({(b'k1',): b'foo', (b'k2',): b'bar', (b'k3',): b'quux'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        child_key = child.serialise(chkmap._store)[0]
        self.assertEqual([child_key, keys[1]], keys)

    def test_map_to_child_child_splits_new(self):
        chkmap = self._get_map({(b'k1',): b'foo', (b'k22',): b'bar'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' LeafNode\n      ('k22',) 'bar'\n", chkmap._dump_tree())
        chkmap = CHKMap(chkmap._store, chkmap._root_node)
        chkmap._ensure_root()
        node = chkmap._root_node
        self.assertEqual(2, len([value for value in node._items.values() if isinstance(value, StaticTuple)]))
        prefix, nodes = node.map(chkmap._store, (b'k23',), b'quux')
        self.assertEqual(b'k', prefix)
        self.assertEqual([(b'', node)], nodes)
        child = node._items[b'k2']
        self.assertIsInstance(child, InternalNode)
        self.assertEqual(2, len(child))
        self.assertEqual({(b'k22',): b'bar', (b'k23',): b'quux'}, self.to_dict(child, None))
        self.assertEqual(None, child._key)
        self.assertEqual(10, child.maximum_size)
        self.assertEqual(1, child._key_width)
        self.assertEqual(3, child._node_width)
        self.assertEqual(3, len(chkmap))
        self.assertEqual({(b'k1',): b'foo', (b'k22',): b'bar', (b'k23',): b'quux'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        child_key = child._key
        k22_key = child._items[b'k22']._key
        k23_key = child._items[b'k23']._key
        self.assertEqual({k22_key, k23_key, child_key, node.key()}, set(keys))
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())

    def test__search_prefix_filter_with_hash(self):
        search_key_func = chk_map.search_key_registry.get(b'hash-16-way')
        node = InternalNode(search_key_func=search_key_func)
        node._key_width = 2
        node._node_width = 4
        self.assertEqual(b'E8B7BE43\x0071BEEFF9', search_key_func(StaticTuple(b'a', b'b')))
        self.assertEqual(b'E8B7', node._search_prefix_filter(StaticTuple(b'a', b'b')))
        self.assertEqual(b'E8B7', node._search_prefix_filter(StaticTuple(b'a')))

    def test_unmap_k23_from_k1_k22_k23_gives_k1_k22_tree_new(self):
        chkmap = self._get_map({(b'k1',): b'foo', (b'k22',): b'bar', (b'k23',): b'quux'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
        chkmap = CHKMap(chkmap._store, chkmap._root_node)
        chkmap._ensure_root()
        node = chkmap._root_node
        result = node.unmap(chkmap._store, (b'k23',))
        child = node._items[b'k2']
        self.assertIsInstance(child, LeafNode)
        self.assertEqual(1, len(child))
        self.assertEqual({(b'k22',): b'bar'}, self.to_dict(child, None))
        self.assertEqual(2, len(chkmap))
        self.assertEqual({(b'k1',): b'foo', (b'k22',): b'bar'}, self.to_dict(chkmap))
        keys = list(node.serialise(chkmap._store))
        self.assertEqual([keys[-1]], keys)
        chkmap = CHKMap(chkmap._store, keys[-1])
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' LeafNode\n      ('k22',) 'bar'\n", chkmap._dump_tree())

    def test_unmap_k1_from_k1_k22_k23_gives_k22_k23_tree_new(self):
        chkmap = self._get_map({(b'k1',): b'foo', (b'k22',): b'bar', (b'k23',): b'quux'}, maximum_size=10)
        self.assertEqualDiff("'' InternalNode\n  'k1' LeafNode\n      ('k1',) 'foo'\n  'k2' InternalNode\n    'k22' LeafNode\n      ('k22',) 'bar'\n    'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
        orig_root = chkmap._root_node
        chkmap = CHKMap(chkmap._store, orig_root)
        chkmap._ensure_root()
        node = chkmap._root_node
        k2_ptr = node._items[b'k2']
        result = node.unmap(chkmap._store, (b'k1',))
        self.assertEqual(k2_ptr, result)
        chkmap = CHKMap(chkmap._store, orig_root)
        chkmap.unmap((b'k1',))
        self.assertEqual(k2_ptr, chkmap._root_node)
        self.assertEqualDiff("'' InternalNode\n  'k22' LeafNode\n      ('k22',) 'bar'\n  'k23' LeafNode\n      ('k23',) 'quux'\n", chkmap._dump_tree())
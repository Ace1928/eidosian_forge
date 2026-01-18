from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
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
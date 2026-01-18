from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_without_fitting_doesnt_page_in(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(20)
    chkmap.map((b'aaa',), b'v')
    chkmap.map((b'aab',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'v'\n", chkmap._dump_tree())
    chkmap = CHKMap(store, chkmap._save())
    chkmap.map((b'aac',), b'v')
    chkmap.map((b'aad',), b'v')
    chkmap.map((b'aae',), b'v')
    chkmap.map((b'aaf',), b'v')
    self.assertIsInstance(chkmap._root_node._items[b'aaa'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aab'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aac'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aad'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aae'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aaf'], LeafNode)
    chkmap.unmap((b'aaf',))
    self.assertIsInstance(chkmap._root_node._items[b'aaa'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aab'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aac'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aad'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aae'], LeafNode)
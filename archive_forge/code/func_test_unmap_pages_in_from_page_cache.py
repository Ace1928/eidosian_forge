from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_pages_in_from_page_cache(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(30)
    chkmap.map((b'aaa',), b'val')
    chkmap.map((b'aab',), b'val')
    chkmap.map((b'aac',), b'val')
    root_key = chkmap._save()
    chkmap = CHKMap(store, root_key)
    chkmap.map((b'aad',), b'val')
    self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'val'\n  'aab' LeafNode\n      ('aab',) 'val'\n  'aac' LeafNode\n      ('aac',) 'val'\n  'aad' LeafNode\n      ('aad',) 'val'\n", chkmap._dump_tree())
    chkmap = CHKMap(store, root_key)
    chkmap.map((b'aad',), b'v')
    self.assertIsInstance(chkmap._root_node._items[b'aaa'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aab'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aac'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aad'], LeafNode)
    aab_key = chkmap._root_node._items[b'aab']
    aab_bytes = chk_map._get_cache()[aab_key]
    aac_key = chkmap._root_node._items[b'aac']
    aac_bytes = chk_map._get_cache()[aac_key]
    chk_map.clear_cache()
    chk_map._get_cache()[aab_key] = aab_bytes
    chk_map._get_cache()[aac_key] = aac_bytes
    chkmap.unmap((b'aad',))
    self.assertIsInstance(chkmap._root_node._items[b'aaa'], StaticTuple)
    self.assertIsInstance(chkmap._root_node._items[b'aab'], LeafNode)
    self.assertIsInstance(chkmap._root_node._items[b'aac'], LeafNode)
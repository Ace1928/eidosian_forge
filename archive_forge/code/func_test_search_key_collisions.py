from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_search_key_collisions(self):
    chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_search_key_single)
    chkmap._root_node.set_maximum_size(20)
    chkmap.map((b'1',), b'foo')
    chkmap.map((b'2',), b'bar')
    chkmap.map((b'3',), b'baz')
    self.assertEqualDiff("'' LeafNode\n      ('1',) 'foo'\n      ('2',) 'bar'\n      ('3',) 'baz'\n", chkmap._dump_tree())
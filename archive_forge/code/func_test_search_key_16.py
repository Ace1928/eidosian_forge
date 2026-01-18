from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_search_key_16(self):
    chk_bytes = self.get_chk_bytes()
    chkmap = chk_map.CHKMap(chk_bytes, None, search_key_func=chk_map._search_key_16)
    chkmap._root_node.set_maximum_size(10)
    chkmap.map((b'1',), b'foo')
    chkmap.map((b'2',), b'bar')
    chkmap.map((b'3',), b'baz')
    self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())
    root_key = chkmap._save()
    chkmap = chk_map.CHKMap(chk_bytes, root_key, search_key_func=chk_map._search_key_16)
    self.assertEqual([((b'1',), b'foo')], list(chkmap.iteritems([(b'1',)])))
    self.assertEqualDiff("'' InternalNode\n  '1' LeafNode\n      ('2',) 'bar'\n  '6' LeafNode\n      ('3',) 'baz'\n  '8' LeafNode\n      ('1',) 'foo'\n", chkmap._dump_tree())
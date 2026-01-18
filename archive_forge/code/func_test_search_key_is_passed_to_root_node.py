from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_search_key_is_passed_to_root_node(self):
    chkmap = chk_map.CHKMap(self.get_chk_bytes(), None, search_key_func=_test_search_key)
    self.assertIs(_test_search_key, chkmap._search_key_func)
    self.assertEqual(b'test:1\x002\x003', chkmap._search_key_func((b'1', b'2', b'3')))
    self.assertEqual(b'test:1\x002\x003', chkmap._root_node._search_key((b'1', b'2', b'3')))
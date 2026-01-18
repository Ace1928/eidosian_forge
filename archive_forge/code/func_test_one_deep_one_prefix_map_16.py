from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_one_prefix_map_16(self):
    c_map = self.make_one_deep_one_prefix_map(search_key_func=chk_map._search_key_16)
    self.assertEqualDiff("'' InternalNode\n  '4' LeafNode\n      ('bbb',) 'initial bbb content'\n  'F' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())
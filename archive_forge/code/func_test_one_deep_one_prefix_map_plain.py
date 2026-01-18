from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_deep_one_prefix_map_plain(self):
    c_map = self.make_one_deep_one_prefix_map()
    self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'b' LeafNode\n      ('bbb',) 'initial bbb content'\n", c_map._dump_tree())
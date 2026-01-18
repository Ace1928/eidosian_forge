from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_root_only_aaa_ddd_plain(self):
    c_map = self.make_root_only_aaa_ddd_map()
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())
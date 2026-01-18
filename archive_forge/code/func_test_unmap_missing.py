from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_missing(self):
    node = LeafNode()
    self.assertRaises(KeyError, node.unmap, None, (b'foo bar',))
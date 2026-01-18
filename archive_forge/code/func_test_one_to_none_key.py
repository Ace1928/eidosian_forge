from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_one_to_none_key(self):
    basis = self.get_map_key({(b'a',): b'content'})
    target = self.get_map_key({})
    self.assertIterInteresting([target], [], [target], [basis])
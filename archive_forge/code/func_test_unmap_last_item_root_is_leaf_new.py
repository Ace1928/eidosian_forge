from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_last_item_root_is_leaf_new(self):
    chkmap = self._get_map({(b'k1' * 50,): b'v1', (b'k2' * 50,): b'v2'})
    chkmap.unmap((b'k1' * 50,))
    chkmap.unmap((b'k2' * 50,))
    self.assertEqual(0, len(chkmap))
    self.assertEqual({}, self.to_dict(chkmap))
    key = chkmap._save()
    leaf_node = LeafNode()
    self.assertEqual([key], leaf_node.serialise(chkmap._store))
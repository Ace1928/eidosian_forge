from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_serialise_empty(self):
    store = self.get_chk_bytes()
    node = LeafNode()
    node.set_maximum_size(10)
    expected_key = (b'sha1:f34c3f0634ea3f85953dffa887620c0a5b1f4a51',)
    self.assertEqual([expected_key], list(node.serialise(store)))
    self.assertEqual(b'chkleaf:\n10\n1\n0\n\n', self.read_bytes(store, expected_key))
    self.assertEqual(expected_key, node.key())
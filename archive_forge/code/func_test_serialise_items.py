from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_serialise_items(self):
    store = self.get_chk_bytes()
    node = LeafNode()
    node.set_maximum_size(10)
    node.map(None, (b'foo bar',), b'baz quux')
    expected_key = (b'sha1:f89fac7edfc6bdb1b1b54a556012ff0c646ef5e0',)
    self.assertEqual(b'foo bar', node._common_serialised_prefix)
    self.assertEqual([expected_key], list(node.serialise(store)))
    self.assertEqual(b'chkleaf:\n10\n1\n1\nfoo bar\n\x001\nbaz quux\n', self.read_bytes(store, expected_key))
    self.assertEqual(expected_key, node.key())
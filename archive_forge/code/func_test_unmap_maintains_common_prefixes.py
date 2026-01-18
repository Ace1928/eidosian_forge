from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_unmap_maintains_common_prefixes(self):
    node = LeafNode()
    node._key_width = 2
    node.map(None, (b'foo bar', b'baz'), b'baz quux')
    node.map(None, (b'foo bar', b'bing'), b'baz quux')
    node.map(None, (b'fool', b'baby'), b'baz quux')
    node.map(None, (b'very', b'different'), b'value')
    self.assertEqual(b'', node._search_prefix)
    self.assertEqual(b'', node._common_serialised_prefix)
    node.unmap(None, (b'very', b'different'))
    self.assertEqual(b'foo', node._search_prefix)
    self.assertEqual(b'foo', node._common_serialised_prefix)
    node.unmap(None, (b'fool', b'baby'))
    self.assertEqual(b'foo bar\x00b', node._search_prefix)
    self.assertEqual(b'foo bar\x00b', node._common_serialised_prefix)
    node.unmap(None, (b'foo bar', b'baz'))
    self.assertEqual(b'foo bar\x00bing', node._search_prefix)
    self.assertEqual(b'foo bar\x00bing', node._common_serialised_prefix)
    node.unmap(None, (b'foo bar', b'bing'))
    self.assertEqual(None, node._search_prefix)
    self.assertEqual(None, node._common_serialised_prefix)
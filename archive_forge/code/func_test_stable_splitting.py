from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def test_stable_splitting(self):
    store = self.get_chk_bytes()
    chkmap = CHKMap(store, None)
    chkmap._root_node.set_maximum_size(35)
    chkmap.map((b'aaa',), b'v')
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n", chkmap._dump_tree())
    chkmap.map((b'aab',), b'v')
    self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'v'\n      ('aab',) 'v'\n", chkmap._dump_tree())
    self.assertCanonicalForm(chkmap)
    chkmap.map((b'aac',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'aaa' LeafNode\n      ('aaa',) 'v'\n  'aab' LeafNode\n      ('aab',) 'v'\n  'aac' LeafNode\n      ('aac',) 'v'\n", chkmap._dump_tree())
    self.assertCanonicalForm(chkmap)
    chkmap.map((b'bbb',), b'v')
    self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aaa' LeafNode\n      ('aaa',) 'v'\n    'aab' LeafNode\n      ('aab',) 'v'\n    'aac' LeafNode\n      ('aac',) 'v'\n  'b' LeafNode\n      ('bbb',) 'v'\n", chkmap._dump_tree())
    self.assertCanonicalForm(chkmap)
from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def assertCanonicalForm(self, chkmap):
    """Assert that the chkmap is in 'canonical' form.

        We do this by adding all of the key value pairs from scratch, both in
        forward order and reverse order, and assert that the final tree layout
        is identical.
        """
    items = list(chkmap.iteritems())
    map_forward = chk_map.CHKMap(None, None)
    map_forward._root_node.set_maximum_size(chkmap._root_node.maximum_size)
    for key, value in items:
        map_forward.map(key, value)
    self.assertMapLayoutEqual(map_forward, chkmap)
    map_reverse = chk_map.CHKMap(None, None)
    map_reverse._root_node.set_maximum_size(chkmap._root_node.maximum_size)
    for key, value in reversed(items):
        map_reverse.map(key, value)
    self.assertMapLayoutEqual(map_reverse, chkmap)
from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def _search_key_single(key):
    """A search key function that maps all nodes to the same value"""
    return 'value'
from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _walk_through_internal_nodes(self, keys):
    """Take the given set of keys, and find the corresponding LeafNodes.

        :param keys: An unsorted iterable of keys to search for
        :return: (nodes, index_and_keys)
            nodes is a dict mapping {index: LeafNode}
            keys_at_index is a list of tuples of [(index, [keys for Leaf])]
        """
    keys_at_index = [(0, sorted(keys))]
    for row_pos, next_row_start in enumerate(self._row_offsets[1:-1]):
        node_indexes = [idx for idx, s_keys in keys_at_index]
        nodes = self._get_internal_nodes(node_indexes)
        next_nodes_and_keys = []
        for node_index, sub_keys in keys_at_index:
            node = nodes[node_index]
            positions = self._multi_bisect_right(sub_keys, node.keys)
            node_offset = next_row_start + node.offset
            next_nodes_and_keys.extend([(node_offset + pos, s_keys) for pos, s_keys in positions])
        keys_at_index = next_nodes_and_keys
    node_indexes = [idx for idx, s_keys in keys_at_index]
    nodes = self._get_leaf_nodes(node_indexes)
    return (nodes, keys_at_index)
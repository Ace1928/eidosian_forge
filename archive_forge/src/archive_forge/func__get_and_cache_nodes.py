from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _get_and_cache_nodes(self, nodes):
    """Read nodes and cache them in the lru.

        The nodes list supplied is sorted and then read from disk, each node
        being inserted it into the _node_cache.

        Note: Asking for more nodes than the _node_cache can contain will
        result in some of the results being immediately discarded, to prevent
        this an assertion is raised if more nodes are asked for than are
        cachable.

        :return: A dict of {node_pos: node}
        """
    found = {}
    start_of_leaves = None
    for node_pos, node in self._read_nodes(sorted(nodes)):
        if node_pos == 0:
            self._root_node = node
        else:
            if start_of_leaves is None:
                start_of_leaves = self._row_offsets[-2]
            if node_pos < start_of_leaves:
                self._internal_node_cache[node_pos] = node
            else:
                self._leaf_node_cache[node_pos] = node
        found[node_pos] = node
    return found
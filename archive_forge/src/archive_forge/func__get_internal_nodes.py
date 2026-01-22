from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _get_internal_nodes(self, node_indexes):
    """Get a node, from cache or disk.

        After getting it, the node will be cached.
        """
    return self._get_nodes(self._internal_node_cache, node_indexes)
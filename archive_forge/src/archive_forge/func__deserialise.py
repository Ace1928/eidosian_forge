import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def _deserialise(data, key, search_key_func):
    """Helper for repositorydetails - convert bytes to a node."""
    if data.startswith(b'chkleaf:\n'):
        node = LeafNode.deserialise(data, key, search_key_func=search_key_func)
    elif data.startswith(b'chknode:\n'):
        node = InternalNode.deserialise(data, key, search_key_func=search_key_func)
    else:
        raise AssertionError('Unknown node type.')
    return node
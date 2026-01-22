import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _similarity_score(obj1, obj2, block_cache=None):
    """Compute a similarity score for two objects.

    Args:
      obj1: The first object to score.
      obj2: The second object to score.
      block_cache: An optional dict of SHA to block counts to cache
        results between calls.

    Returns:
      The similarity score between the two objects, defined as the
        number of bytes in common between the two objects divided by the
        maximum size, scaled to the range 0-100.
    """
    if block_cache is None:
        block_cache = {}
    if obj1.id not in block_cache:
        block_cache[obj1.id] = _count_blocks(obj1)
    if obj2.id not in block_cache:
        block_cache[obj2.id] = _count_blocks(obj2)
    common_bytes = _common_bytes(block_cache[obj1.id], block_cache[obj2.id])
    max_size = max(obj1.raw_length(), obj2.raw_length())
    if not max_size:
        return _MAX_SCORE
    return int(float(common_bytes) * _MAX_SCORE / max_size)
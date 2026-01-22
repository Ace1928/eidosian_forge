import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def _common_bytes(blocks1, blocks2):
    """Count the number of common bytes in two block count dicts.

    Args:
      blocks1: The first dict of block hashcode -> total bytes.
      blocks2: The second dict of block hashcode -> total bytes.

    Returns:
      The number of bytes in common between blocks1 and blocks2. This is
      only approximate due to possible hash collisions.
    """
    if len(blocks1) > len(blocks2):
        blocks1, blocks2 = (blocks2, blocks1)
    score = 0
    for block, count1 in blocks1.items():
        count2 = blocks2.get(block)
        if count2:
            score += min(count1, count2)
    return score
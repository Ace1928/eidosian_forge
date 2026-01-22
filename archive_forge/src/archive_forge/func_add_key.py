import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
def add_key(self, key):
    """Add another to key to fetch.

        :return: The estimated number of bytes needed to fetch the batch so
            far.
        """
    self.keys.append(key)
    index_memo, _, _, _ = self.locations[key]
    read_memo = index_memo[0:3]
    if read_memo in self.batch_memos:
        return self.total_bytes
    try:
        cached_block = self.gcvf._group_cache[read_memo]
    except KeyError:
        self.batch_memos[read_memo] = None
        self.memos_to_get.append(read_memo)
        byte_length = read_memo[2]
        self.total_bytes += byte_length
    else:
        self.batch_memos[read_memo] = cached_block
    return self.total_bytes
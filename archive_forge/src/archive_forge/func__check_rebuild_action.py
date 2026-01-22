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
def _check_rebuild_action(self):
    """Check to see if our block should be repacked."""
    total_bytes_used = 0
    last_byte_used = 0
    for factory in self._factories:
        total_bytes_used += factory._end - factory._start
        if last_byte_used < factory._end:
            last_byte_used = factory._end
    if total_bytes_used * 2 >= self._block._content_length:
        return (None, last_byte_used, total_bytes_used)
    if total_bytes_used * 2 > last_byte_used:
        return ('trim', last_byte_used, total_bytes_used)
    return ('rebuild', last_byte_used, total_bytes_used)
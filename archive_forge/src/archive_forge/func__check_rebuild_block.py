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
def _check_rebuild_block(self):
    action, last_byte_used, total_bytes_used = self._check_rebuild_action()
    if action is None:
        return
    if action == 'trim':
        self._trim_block(last_byte_used)
    elif action == 'rebuild':
        self._rebuild_block()
    else:
        raise ValueError('unknown rebuild action: {!r}'.format(action))
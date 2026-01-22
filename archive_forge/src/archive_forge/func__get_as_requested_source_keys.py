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
def _get_as_requested_source_keys(self, orig_keys, locations, unadded_keys, key_to_source_map):
    source_keys = []
    current_source = None
    for key in orig_keys:
        if key in locations or key in unadded_keys:
            source = self
        elif key in key_to_source_map:
            source = key_to_source_map[key]
        else:
            continue
        if source is not current_source:
            source_keys.append((source, []))
            current_source = source
        source_keys[-1][1].append(key)
    return source_keys
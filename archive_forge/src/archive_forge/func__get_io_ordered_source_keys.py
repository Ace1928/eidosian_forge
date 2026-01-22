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
def _get_io_ordered_source_keys(self, locations, unadded_keys, source_result):

    def get_group(key):
        return locations[key][0]
    present_keys = list(unadded_keys)
    present_keys.extend(sorted(locations, key=get_group))
    source_keys = [(self, present_keys)]
    source_keys.extend(source_result)
    return source_keys
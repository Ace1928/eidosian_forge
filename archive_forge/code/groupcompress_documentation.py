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
Inform this _GCGraphIndex that there is an unvalidated index.

        This allows this _GCGraphIndex to keep track of any missing
        compression parents we may want to have filled in to make those
        indices valid.  It also allows _GCGraphIndex to track any new keys.

        :param graph_index: A GraphIndex
        
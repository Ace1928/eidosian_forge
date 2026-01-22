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
class _GCBuildDetails:
    """A blob of data about the build details.

    This stores the minimal data, which then allows compatibility with the old
    api, without taking as much memory.
    """
    __slots__ = ('_index', '_group_start', '_group_end', '_basis_end', '_delta_end', '_parents')
    method = 'group'
    compression_parent = None

    def __init__(self, parents, position_info):
        self._parents = parents
        self._index, self._group_start, self._group_end, self._basis_end, self._delta_end = position_info

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.index_memo, self._parents)

    @property
    def index_memo(self):
        return (self._index, self._group_start, self._group_end, self._basis_end, self._delta_end)

    @property
    def record_details(self):
        return static_tuple.StaticTuple(self.method, None)

    def __getitem__(self, offset):
        """Compatibility thunk to act like a tuple."""
        if offset == 0:
            return self.index_memo
        elif offset == 1:
            return self.compression_parent
        elif offset == 2:
            return self._parents
        elif offset == 3:
            return self.record_details
        else:
            raise IndexError('offset out of range')

    def __len__(self):
        return 4
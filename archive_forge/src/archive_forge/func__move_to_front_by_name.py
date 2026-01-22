import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _move_to_front_by_name(self, hit_names):
    """Moves indices named by 'hit_names' to front of the search order, as
        described in _move_to_front.
        """
    indices_info = zip(self._index_names, self._indices)
    hit_indices = []
    for name, idx in indices_info:
        if name in hit_names:
            hit_indices.append(idx)
    self._move_to_front_by_index(hit_indices)
import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
@staticmethod
def _find_index(range_map, key):
    """Helper for the _parsed_*_index calls.

        Given a range map - [(start, end), ...], finds the index of the range
        in the map for key if it is in the map, and if it is not there, the
        immediately preceeding range in the map.
        """
    result = bisect_right(range_map, key) - 1
    if result + 1 < len(range_map):
        if range_map[result + 1][0] == key[0]:
            return result + 1
    return result
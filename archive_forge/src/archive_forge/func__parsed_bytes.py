import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _parsed_bytes(self, start, start_key, end, end_key):
    """Mark the bytes from start to end as parsed.

        Calling self._parsed_bytes(1,2) will mark one byte (the one at offset
        1) as parsed.

        :param start: The start of the parsed region.
        :param end: The end of the parsed region.
        """
    index = self._parsed_byte_index(start)
    new_value = (start, end)
    new_key = (start_key, end_key)
    if index == -1:
        self._parsed_byte_map.insert(index, new_value)
        self._parsed_key_map.insert(index, new_key)
        return
    if index + 1 < len(self._parsed_byte_map) and self._parsed_byte_map[index][1] == start and (self._parsed_byte_map[index + 1][0] == end):
        self._parsed_byte_map[index] = (self._parsed_byte_map[index][0], self._parsed_byte_map[index + 1][1])
        self._parsed_key_map[index] = (self._parsed_key_map[index][0], self._parsed_key_map[index + 1][1])
        del self._parsed_byte_map[index + 1]
        del self._parsed_key_map[index + 1]
    elif self._parsed_byte_map[index][1] == start:
        self._parsed_byte_map[index] = (self._parsed_byte_map[index][0], end)
        self._parsed_key_map[index] = (self._parsed_key_map[index][0], end_key)
    elif index + 1 < len(self._parsed_byte_map) and self._parsed_byte_map[index + 1][0] == end:
        self._parsed_byte_map[index + 1] = (start, self._parsed_byte_map[index + 1][1])
        self._parsed_key_map[index + 1] = (start_key, self._parsed_key_map[index + 1][1])
    else:
        self._parsed_byte_map.insert(index + 1, new_value)
        self._parsed_key_map.insert(index + 1, new_key)
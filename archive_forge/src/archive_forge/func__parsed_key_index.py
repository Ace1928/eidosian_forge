import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _parsed_key_index(self, key):
    """Return the index of the entry immediately before key.

        e.g. if the parsed map has regions (None, 'a') and ('b','c') parsed,
        meaning that keys from None to 'a' inclusive, and 'b' to 'c' inclusive
        have been parsed, then:
        asking for '' will return 0
        asking for 'a' will return 0
        asking for 'b' will return 1
        asking for 'e' will return 1
        """
    search_key = (key, b'')
    return self._find_index(self._parsed_key_map, search_key)
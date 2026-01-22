import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _read_and_parse(self, readv_ranges):
    """Read the ranges and parse the resulting data.

        :param readv_ranges: A prepared readv range list.
        """
    if not readv_ranges:
        return
    if self._nodes is None and self._bytes_read * 2 >= self._size:
        self._buffer_all()
        return
    base_offset = self._base_offset
    if base_offset != 0:
        readv_ranges = [(start + base_offset, size) for start, size in readv_ranges]
    readv_data = self._transport.readv(self._name, readv_ranges, True, self._size + self._base_offset)
    for offset, data in readv_data:
        offset -= base_offset
        self._bytes_read += len(data)
        if offset < 0:
            data = data[-offset:]
            offset = 0
        if offset == 0 and len(data) == self._size:
            self._buffer_all(BytesIO(data))
            return
        if self._bisect_nodes is None:
            if not offset == 0:
                raise AssertionError()
            offset, data = self._parse_header_from_bytes(data)
        self._parse_region(offset, data)
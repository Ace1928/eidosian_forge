import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _parse_segment(self, offset, data, end, index):
    """Parse one segment of data.

        :param offset: Where 'data' begins in the file.
        :param data: Some data to parse a segment of.
        :param end: Where data ends
        :param index: The current index into the parsed bytes map.
        :return: True if the parsed segment is the last possible one in the
            range of data.
        :return: high_parsed_byte, last_segment.
            high_parsed_byte is the location of the highest parsed byte in this
            segment, last_segment is True if the parsed segment is the last
            possible one in the data block.
        """
    trim_end = None
    if offset < self._parsed_byte_map[index][1]:
        trim_start = self._parsed_byte_map[index][1] - offset
        start_adjacent = True
    elif offset == self._parsed_byte_map[index][1]:
        trim_start = None
        start_adjacent = True
    else:
        trim_start = None
        start_adjacent = False
    if end == self._size:
        trim_end = None
        end_adjacent = True
        last_segment = True
    elif index + 1 == len(self._parsed_byte_map):
        trim_end = None
        end_adjacent = False
        last_segment = True
    elif end == self._parsed_byte_map[index + 1][0]:
        trim_end = None
        end_adjacent = True
        last_segment = True
    elif end > self._parsed_byte_map[index + 1][0]:
        trim_end = self._parsed_byte_map[index + 1][0] - offset
        end_adjacent = True
        last_segment = end < self._parsed_byte_map[index + 1][1]
    else:
        trim_end = None
        end_adjacent = False
        last_segment = True
    if not start_adjacent:
        if trim_start is None:
            trim_start = data.find(b'\n') + 1
        else:
            trim_start = data.find(b'\n', trim_start) + 1
        if not trim_start != 0:
            raise AssertionError('no \n was present')
    if not end_adjacent:
        if trim_end is None:
            trim_end = data.rfind(b'\n') + 1
        else:
            trim_end = data.rfind(b'\n', None, trim_end) + 1
        if not trim_end != 0:
            raise AssertionError('no \n was present')
    trimmed_data = data[trim_start:trim_end]
    if not trimmed_data:
        raise AssertionError('read unneeded data [%d:%d] from [%d:%d]' % (trim_start, trim_end, offset, offset + len(data)))
    if trim_start:
        offset += trim_start
    lines = trimmed_data.split(b'\n')
    del lines[-1]
    pos = offset
    first_key, last_key, nodes, _ = self._parse_lines(lines, pos)
    for key, value in nodes:
        self._bisect_nodes[key] = value
    self._parsed_bytes(offset, first_key, offset + len(trimmed_data), last_key)
    return (offset + len(trimmed_data), last_segment)
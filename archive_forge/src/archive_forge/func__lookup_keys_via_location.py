import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _lookup_keys_via_location(self, location_keys):
    """Public interface for implementing bisection.

        If _buffer_all has been called, then all the data for the index is in
        memory, and this method should not be called, as it uses a separate
        cache because it cannot pre-resolve all indices, which buffer_all does
        for performance.

        :param location_keys: A list of location(byte offset), key tuples.
        :return: A list of (location_key, result) tuples as expected by
            breezy.bisect_multi.bisect_multi_bytes.
        """
    readv_ranges = []
    for location, key in location_keys:
        if self._bisect_nodes and key in self._bisect_nodes:
            continue
        index = self._parsed_key_index(key)
        if len(self._parsed_key_map) and self._parsed_key_map[index][0] <= key and (self._parsed_key_map[index][1] >= key or self._parsed_byte_map[index][1] == self._size):
            continue
        index = self._parsed_byte_index(location)
        if len(self._parsed_byte_map) and self._parsed_byte_map[index][0] <= location and (self._parsed_byte_map[index][1] > location):
            continue
        length = 800
        if location + length > self._size:
            length = self._size - location
        if length > 0:
            readv_ranges.append((location, length))
    if self._bisect_nodes is None:
        readv_ranges.append(_HEADER_READV)
    self._read_and_parse(readv_ranges)
    result = []
    if self._nodes is not None:
        for location, key in location_keys:
            if key not in self._nodes:
                result.append(((location, key), False))
            elif self.node_ref_lists:
                value, refs = self._nodes[key]
                result.append(((location, key), (self, key, value, refs)))
            else:
                result.append(((location, key), (self, key, self._nodes[key])))
        return result
    pending_references = []
    pending_locations = set()
    for location, key in location_keys:
        if key in self._bisect_nodes:
            if self.node_ref_lists:
                value, refs = self._bisect_nodes[key]
                wanted_locations = []
                for ref_list in refs:
                    for ref in ref_list:
                        if ref not in self._keys_by_offset:
                            wanted_locations.append(ref)
                if wanted_locations:
                    pending_locations.update(wanted_locations)
                    pending_references.append((location, key))
                    continue
                result.append(((location, key), (self, key, value, self._resolve_references(refs))))
            else:
                result.append(((location, key), (self, key, self._bisect_nodes[key])))
            continue
        else:
            index = self._parsed_key_index(key)
            if self._parsed_key_map[index][0] <= key and (self._parsed_key_map[index][1] >= key or self._parsed_byte_map[index][1] == self._size):
                result.append(((location, key), False))
                continue
        index = self._parsed_byte_index(location)
        if key < self._parsed_key_map[index][0]:
            direction = -1
        else:
            direction = +1
        result.append(((location, key), direction))
    readv_ranges = []
    for location in pending_locations:
        length = 800
        if location + length > self._size:
            length = self._size - location
        if length > 0:
            readv_ranges.append((location, length))
    self._read_and_parse(readv_ranges)
    if self._nodes is not None:
        for location, key in pending_references:
            value, refs = self._nodes[key]
            result.append(((location, key), (self, key, value, refs)))
        return result
    for location, key in pending_references:
        value, refs = self._bisect_nodes[key]
        result.append(((location, key), (self, key, value, self._resolve_references(refs))))
    return result
from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _read_nodes(self, nodes):
    """Read some nodes from disk into the LRU cache.

        This performs a readv to get the node data into memory, and parses each
        node, then yields it to the caller. The nodes are requested in the
        supplied order. If possible doing sort() on the list before requesting
        a read may improve performance.

        :param nodes: The nodes to read. 0 - first node, 1 - second node etc.
        :return: None
        """
    bytes = None
    ranges = []
    base_offset = self._base_offset
    for index in nodes:
        offset = index * _PAGE_SIZE
        size = _PAGE_SIZE
        if index == 0:
            if self._size:
                size = min(_PAGE_SIZE, self._size)
            else:
                bytes = self._transport.get_bytes(self._name)
                num_bytes = len(bytes)
                self._size = num_bytes - base_offset
                ranges = [(start, min(_PAGE_SIZE, num_bytes - start)) for start in range(base_offset, num_bytes, _PAGE_SIZE)]
                break
        else:
            if offset > self._size:
                raise AssertionError('tried to read past the end of the file %s > %s' % (offset, self._size))
            size = min(size, self._size - offset)
        ranges.append((base_offset + offset, size))
    if not ranges:
        return
    elif bytes is not None:
        data_ranges = [(start, bytes[start:start + size]) for start, size in ranges]
    elif self._file is None:
        data_ranges = self._transport.readv(self._name, ranges)
    else:
        data_ranges = []
        for offset, size in ranges:
            self._file.seek(offset)
            data_ranges.append((offset, self._file.read(size)))
    for offset, data in data_ranges:
        offset -= base_offset
        if offset == 0:
            offset, data = self._parse_header_from_bytes(data)
            if len(data) == 0:
                continue
        bytes = zlib.decompress(data)
        if bytes.startswith(_LEAF_FLAG):
            node = self._leaf_factory(bytes, self._key_length, self.node_ref_lists)
        elif bytes.startswith(_INTERNAL_FLAG):
            node = _InternalNode(bytes)
        else:
            raise AssertionError('Unknown node type for %r' % bytes)
        yield (offset // _PAGE_SIZE, node)
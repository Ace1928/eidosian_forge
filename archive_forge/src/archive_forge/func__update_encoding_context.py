import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _update_encoding_context(self, data):
    """
        Handles a byte that updates the encoding context.
        """
    new_size, consumed = decode_integer(data, 5)
    if new_size > self.max_allowed_table_size:
        raise InvalidTableSizeError('Encoder exceeded max allowable table size')
    self.header_table_size = new_size
    return consumed
import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _encode_literal(self, name, value, indexbit, huffman=False):
    """
        Encodes a header with a literal name and literal value. If ``indexing``
        is True, the header will be added to the header table: otherwise it
        will not.
        """
    if huffman:
        name = self.huffman_coder.encode(name)
        value = self.huffman_coder.encode(value)
    name_len = encode_integer(len(name), 7)
    value_len = encode_integer(len(value), 7)
    if huffman:
        name_len[0] |= 128
        value_len[0] |= 128
    return b''.join([indexbit, bytes(name_len), name, bytes(value_len), value])
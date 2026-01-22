import logging
from .table import HeaderTable, table_entry_size
from .compat import to_byte, to_bytes
from .exceptions import (
from .huffman import HuffmanEncoder
from .huffman_constants import (
from .huffman_table import decode_huffman
from .struct import HeaderTuple, NeverIndexedHeaderTuple
def _dict_to_iterable(header_dict):
    """
    This converts a dictionary to an iterable of two-tuples. This is a
    HPACK-specific function becuase it pulls "special-headers" out first and
    then emits them.
    """
    assert isinstance(header_dict, dict)
    keys = sorted(header_dict.keys(), key=lambda k: not _to_bytes(k).startswith(b':'))
    for key in keys:
        yield (key, header_dict[key])
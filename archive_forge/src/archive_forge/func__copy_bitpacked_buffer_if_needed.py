from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _copy_bitpacked_buffer_if_needed(buf: 'pyarrow.Buffer', offset: int, length: int) -> 'pyarrow.Buffer':
    """Copy bit-packed binary buffer, if needed."""
    bit_offset = offset % 8
    byte_offset = offset // 8
    byte_length = _bytes_for_bits(bit_offset + length) // 8
    if offset > 0 or byte_length < buf.size:
        buf = buf.slice(byte_offset, byte_length)
        if bit_offset != 0:
            buf = _align_bit_offset(buf, bit_offset, byte_length)
    return buf
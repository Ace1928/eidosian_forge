from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _copy_normal_buffer_if_needed(buf: 'pyarrow.Buffer', byte_width: int, offset: int, length: int) -> 'pyarrow.Buffer':
    """Copy buffer, if needed."""
    byte_offset = offset * byte_width
    byte_length = length * byte_width
    if offset > 0 or byte_length < buf.size:
        buf = buf.slice(byte_offset, byte_length)
    return buf
from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _copy_offsets_buffer_if_needed(buf: 'pyarrow.Buffer', arr_type: 'pyarrow.DataType', offset: int, length: int) -> Tuple['pyarrow.Buffer', int, int]:
    """Copy the provided offsets buffer, returning the copied buffer and the
    offset + length of the underlying data.
    """
    import pyarrow as pa
    import pyarrow.compute as pac
    if pa.types.is_large_list(arr_type) or pa.types.is_large_string(arr_type) or pa.types.is_large_binary(arr_type) or pa.types.is_large_unicode(arr_type):
        offset_type = pa.int64()
    else:
        offset_type = pa.int32()
    buf = _copy_buffer_if_needed(buf, offset_type, offset, length + 1)
    offsets = pa.Array.from_buffers(offset_type, length + 1, [None, buf])
    child_offset = offsets[0].as_py()
    child_length = offsets[-1].as_py() - child_offset
    offsets = pac.subtract(offsets, child_offset)
    if pa.types.is_int32(offset_type):
        offsets = offsets.cast(offset_type, safe=False)
    buf = offsets.buffers()[1]
    return (buf, child_offset, child_length)
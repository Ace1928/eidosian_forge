from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _binary_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    """Serialize binary (variable-sized binary, string) arrays to
    PicklableArrayPayload.
    """
    assert _is_binary(a.type), a.type
    buffers = a.buffers()
    assert len(buffers) == 3, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    offset_buf = buffers[1]
    offset_buf, data_offset, data_length = _copy_offsets_buffer_if_needed(offset_buf, a.type, a.offset, len(a))
    data_buf = buffers[2]
    data_buf = _copy_buffer_if_needed(data_buf, None, data_offset, data_length)
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, offset_buf, data_buf], null_count=a.null_count, offset=0, children=[])
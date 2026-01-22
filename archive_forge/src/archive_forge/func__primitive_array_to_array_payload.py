from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _primitive_array_to_array_payload(a: 'pyarrow.Array') -> 'PicklableArrayPayload':
    """Serialize primitive (numeric, temporal, boolean) arrays to
    PicklableArrayPayload.
    """
    assert _is_primitive(a.type), a.type
    buffers = a.buffers()
    assert len(buffers) == 2, len(buffers)
    bitmap_buf = buffers[0]
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(bitmap_buf, a.offset, len(a))
    else:
        bitmap_buf = None
    data_buf = buffers[1]
    if data_buf is not None:
        data_buf = _copy_buffer_if_needed(buffers[1], a.type, a.offset, len(a))
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, data_buf], null_count=a.null_count, offset=0, children=[])
from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _struct_array_to_array_payload(a: 'pyarrow.StructArray') -> 'PicklableArrayPayload':
    """Serialize struct arrays to PicklableArrayPayload."""
    buffers = a.buffers()
    assert len(buffers) >= 1, len(buffers)
    if a.null_count > 0:
        bitmap_buf = _copy_bitpacked_buffer_if_needed(buffers[0], a.offset, len(a))
    else:
        bitmap_buf = None
    children = [_array_to_array_payload(a.field(i)) for i in range(a.type.num_fields)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf], null_count=a.null_count, offset=0, children=children)
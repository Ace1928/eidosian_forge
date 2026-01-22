from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _union_array_to_array_payload(a: 'pyarrow.UnionArray') -> 'PicklableArrayPayload':
    """Serialize union arrays to PicklableArrayPayload."""
    import pyarrow as pa
    assert not _is_dense_union(a.type)
    buffers = a.buffers()
    assert len(buffers) > 1, len(buffers)
    bitmap_buf = buffers[0]
    assert bitmap_buf is None, bitmap_buf
    type_code_buf = buffers[1]
    type_code_buf = _copy_buffer_if_needed(type_code_buf, pa.int8(), a.offset, len(a))
    children = [_array_to_array_payload(a.field(i)) for i in range(a.type.num_fields)]
    return PicklableArrayPayload(type=a.type, length=len(a), buffers=[bitmap_buf, type_code_buf], null_count=a.null_count, offset=0, children=children)
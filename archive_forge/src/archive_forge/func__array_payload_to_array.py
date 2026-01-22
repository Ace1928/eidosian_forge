from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _array_payload_to_array(payload: 'PicklableArrayPayload') -> 'pyarrow.Array':
    """Reconstruct an Arrow Array from a possibly nested PicklableArrayPayload."""
    import pyarrow as pa
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType, ArrowVariableShapedTensorType
    children = [child_payload.to_array() for child_payload in payload.children]
    if pa.types.is_dictionary(payload.type):
        assert len(children) == 2, len(children)
        indices, dictionary = children
        return pa.DictionaryArray.from_arrays(indices, dictionary)
    elif pa.types.is_map(payload.type) and len(children) > 1:
        assert len(children) == 3, len(children)
        offsets, keys, items = children
        return pa.MapArray.from_arrays(offsets, keys, items)
    elif isinstance(payload.type, ArrowTensorType) or isinstance(payload.type, ArrowVariableShapedTensorType):
        assert len(children) == 1, len(children)
        storage = children[0]
        return pa.ExtensionArray.from_storage(payload.type, storage)
    else:
        return pa.Array.from_buffers(type=payload.type, length=payload.length, buffers=payload.buffers, null_count=payload.null_count, offset=payload.offset, children=children)
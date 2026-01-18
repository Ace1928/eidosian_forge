from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def validity_buffer_from_mask(validity_buff: BufferObject, validity_dtype: Dtype, describe_null: ColumnNullType, length: int, offset: int=0, allow_copy: bool=True) -> pa.Buffer:
    """
    Build a PyArrow buffer from the passed mask buffer.

    Parameters
    ----------
    validity_buff : BufferObject
        Tuple of underlying validity buffer and associated dtype.
    validity_dtype : Dtype
        Dtype description as a tuple ``(kind, bit-width, format string,
        endianness)``.
    describe_null : ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    length : int
        The number of values in the array.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Buffer
    """
    null_kind, sentinel_val = describe_null
    validity_kind, _, _, _ = validity_dtype
    assert validity_kind == DtypeKind.BOOL
    if null_kind == ColumnNullType.NON_NULLABLE:
        return None
    elif null_kind == ColumnNullType.USE_BYTEMASK or (null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 1):
        buff = pa.foreign_buffer(validity_buff.ptr, validity_buff.bufsize, base=validity_buff)
        if null_kind == ColumnNullType.USE_BYTEMASK:
            if not allow_copy:
                raise RuntimeError('To create a bitmask a copy of the data is required which is forbidden by allow_copy=False')
            mask = pa.Array.from_buffers(pa.int8(), length, [None, buff], offset=offset)
            mask_bool = pc.cast(mask, pa.bool_())
        else:
            mask_bool = pa.Array.from_buffers(pa.bool_(), length, [None, buff], offset=offset)
        if sentinel_val == 1:
            mask_bool = pc.invert(mask_bool)
        return mask_bool.buffers()[1]
    elif null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 0:
        return pa.foreign_buffer(validity_buff.ptr, validity_buff.bufsize, base=validity_buff)
    else:
        raise NotImplementedError(f'{describe_null} null representation is not yet supported.')
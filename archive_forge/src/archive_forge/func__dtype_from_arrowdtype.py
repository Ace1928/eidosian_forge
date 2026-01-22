from __future__ import annotations
import enum
from typing import (
import sys
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.interchange.buffer import _PyArrowBuffer
def _dtype_from_arrowdtype(self, dtype: pa.DataType, bit_width: int) -> Tuple[DtypeKind, int, str, str]:
    """
        See `self.dtype` for details.
        """
    if pa.types.is_timestamp(dtype):
        kind = DtypeKind.DATETIME
        ts = dtype.unit[0]
        tz = dtype.tz if dtype.tz else ''
        f_string = 'ts{ts}:{tz}'.format(ts=ts, tz=tz)
        return (kind, bit_width, f_string, Endianness.NATIVE)
    elif pa.types.is_dictionary(dtype):
        kind = DtypeKind.CATEGORICAL
        arr = self._col
        indices_dtype = arr.indices.type
        _, f_string = _PYARROW_KINDS.get(indices_dtype)
        return (kind, bit_width, f_string, Endianness.NATIVE)
    else:
        kind, f_string = _PYARROW_KINDS.get(dtype, (None, None))
        if kind is None:
            raise ValueError(f'Data type {dtype} not supported by interchange protocol')
        return (kind, bit_width, f_string, Endianness.NATIVE)
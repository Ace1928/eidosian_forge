from __future__ import annotations
from typing import Any
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def _get_offsets_buffer(self) -> tuple[PandasBuffer, Any]:
    """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
    if self.dtype[0] == DtypeKind.STRING:
        values = self._col.to_numpy()
        ptr = 0
        offsets = np.zeros(shape=(len(values) + 1,), dtype=np.int64)
        for i, v in enumerate(values):
            if isinstance(v, str):
                b = v.encode(encoding='utf-8')
                ptr += len(b)
            offsets[i + 1] = ptr
        buffer = PandasBuffer(offsets)
        dtype = (DtypeKind.INT, 64, ArrowCTypes.INT64, Endianness.NATIVE)
    else:
        raise NoBufferPresent('This column has a fixed-length dtype so it does not have an offsets buffer')
    return (buffer, dtype)
import ctypes
import re
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def _inverse_null_buf(buf: np.ndarray, null_kind: ColumnNullType) -> np.ndarray:
    """
    Inverse the boolean value of buffer storing either bit- or bytemask.

    Parameters
    ----------
    buf : np.ndarray
        Buffer to inverse the boolean value for.
    null_kind : {ColumnNullType.USE_BYTEMASK, ColumnNullType.USE_BITMASK}
        How to treat the buffer.

    Returns
    -------
    np.ndarray
        Logically inversed buffer.
    """
    if null_kind == ColumnNullType.USE_BITMASK:
        return ~buf
    assert null_kind == ColumnNullType.USE_BYTEMASK, f'Unexpected null kind: {null_kind}'
    return buf == 0
import ctypes
import re
from typing import Any, Optional, Tuple, Union
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
def unpack_protocol_column(col: ProtocolColumn) -> Tuple[Union[np.ndarray, pandas.Series], Any]:
    """
    Unpack an interchange protocol column to a pandas-ready column.

    Parameters
    ----------
    col : ProtocolColumn
        Column to unpack.

    Returns
    -------
    tuple
        Tuple of resulting column (either an ndarray or a series) and the object
        which keeps memory referenced by the column alive.
    """
    dtype = col.dtype[0]
    if dtype in (DTypeKind.INT, DTypeKind.UINT, DTypeKind.FLOAT, DTypeKind.BOOL):
        return primitive_column_to_ndarray(col)
    elif dtype == DTypeKind.CATEGORICAL:
        return categorical_column_to_series(col)
    elif dtype == DTypeKind.STRING:
        return string_column_to_ndarray(col)
    elif dtype == DTypeKind.DATETIME:
        return datetime_column_to_ndarray(col)
    else:
        raise NotImplementedError(f'Data type {dtype} not handled yet')
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.utils import _inherit_docstrings
from .buffer import PandasProtocolBuffer
from .exception import NoOffsetsBuffer, NoValidityBuffer
def _dtype_from_primitive_pandas_dtype(self, dtype) -> Tuple[DTypeKind, int, str, str]:
    """
        Deduce dtype specific for the protocol from pandas dtype.

        See `self.dtype` for details.

        Parameters
        ----------
        dtype : any
            A pandas dtype.

        Returns
        -------
        tuple
        """
    _np_kinds = {'i': DTypeKind.INT, 'u': DTypeKind.UINT, 'f': DTypeKind.FLOAT, 'b': DTypeKind.BOOL}
    kind = _np_kinds.get(dtype.kind, None)
    if kind is None:
        raise NotImplementedError(f'Data type {dtype} not supported by the dataframe exchange protocol')
    return (kind, dtype.itemsize * 8, pandas_dtype_to_arrow_c(dtype), dtype.byteorder)
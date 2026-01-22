from math import ceil
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas
import pyarrow as pa
from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
from modin.utils import _inherit_docstrings
from .buffer import HdkProtocolBuffer
from .utils import arrow_dtype_to_arrow_c, arrow_types_map
def _propagate_dtype(self, dtype: Tuple[DTypeKind, int, str, str]):
    """
        Propagate `dtype` to the underlying PyArrow table.

        Modifies the column object inplace by replacing underlying PyArrow table with
        the casted one.

        Parameters
        ----------
        dtype : tuple
            Data type conforming protocol dtypes format to cast underlying PyArrow table.
        """
    if not self._col._allow_copy:
        raise_copy_alert(copy_reason='casting to align pandas and PyArrow data types')
    kind, bit_width, format_str, _ = dtype
    arrow_type = None
    if kind in arrow_types_map:
        arrow_type = arrow_types_map[kind].get(bit_width, None)
    elif kind == DTypeKind.DATETIME:
        arrow_type = pa.timestamp('ns')
    elif kind == DTypeKind.CATEGORICAL:
        index_type = arrow_types_map[DTypeKind.INT].get(bit_width, None)
        if index_type is not None:
            arrow_type = pa.dictionary(index_type=index_type, value_type=pa.string())
    if arrow_type is None:
        raise NotImplementedError(f'Propagation for type {dtype} is not supported.')
    at = self._pyarrow_table
    schema_to_cast = at.schema
    field = at.schema[-1]
    schema_to_cast = schema_to_cast.set(len(schema_to_cast) - 1, pa.field(field.name, arrow_type, field.nullable))
    self._cast_at(schema_to_cast)
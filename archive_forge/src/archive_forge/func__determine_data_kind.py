from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def _determine_data_kind(column: Series | Index, field: pa.Field | None=None) -> ColumnDataKind:
    """Determine the data kind of a column.

    The column data kind refers to the shared data type of the values
    in the column (e.g. int, float, str, bool).

    Parameters
    ----------
    column : pd.Series, pd.Index
        The column to determine the data kind for.
    field : pa.Field, optional
        The arrow field from the arrow table schema.

    Returns
    -------
    ColumnDataKind
        The data kind of the column.
    """
    import pandas as pd
    if isinstance(column.dtype, pd.CategoricalDtype):
        return _determine_data_kind_via_inferred_type(column.dtype.categories)
    if field is not None:
        data_kind = _determine_data_kind_via_arrow(field)
        if data_kind != ColumnDataKind.UNKNOWN:
            return data_kind
    if column.dtype.name == 'object':
        return _determine_data_kind_via_inferred_type(column)
    return _determine_data_kind_via_pandas_dtype(column)
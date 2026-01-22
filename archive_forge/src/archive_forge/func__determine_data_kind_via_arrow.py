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
def _determine_data_kind_via_arrow(field: pa.Field) -> ColumnDataKind:
    """Determine the data kind via the arrow type information.

    The column data kind refers to the shared data type of the values
    in the column (e.g. int, float, str, bool).

    Parameters
    ----------

    field : pa.Field
        The arrow field from the arrow table schema.

    Returns
    -------
    ColumnDataKind
        The data kind of the field.
    """
    import pyarrow as pa
    field_type = field.type
    if pa.types.is_integer(field_type):
        return ColumnDataKind.INTEGER
    if pa.types.is_floating(field_type):
        return ColumnDataKind.FLOAT
    if pa.types.is_boolean(field_type):
        return ColumnDataKind.BOOLEAN
    if pa.types.is_string(field_type):
        return ColumnDataKind.STRING
    if pa.types.is_date(field_type):
        return ColumnDataKind.DATE
    if pa.types.is_time(field_type):
        return ColumnDataKind.TIME
    if pa.types.is_timestamp(field_type):
        return ColumnDataKind.DATETIME
    if pa.types.is_duration(field_type):
        return ColumnDataKind.TIMEDELTA
    if pa.types.is_list(field_type):
        return ColumnDataKind.LIST
    if pa.types.is_decimal(field_type):
        return ColumnDataKind.DECIMAL
    if pa.types.is_null(field_type):
        return ColumnDataKind.EMPTY
    if pa.types.is_binary(field_type):
        return ColumnDataKind.BYTES
    if pa.types.is_struct(field_type):
        return ColumnDataKind.DICT
    return ColumnDataKind.UNKNOWN
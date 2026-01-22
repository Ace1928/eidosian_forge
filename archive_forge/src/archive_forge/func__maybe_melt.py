from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
def _maybe_melt(df: pd.DataFrame, x_column: str | None, y_column_list: list[str], color_column: str | None, size_column: str | None) -> tuple[pd.DataFrame, str | None, str | None]:
    """If multiple columns are set for y, melt the dataframe into long format."""
    y_column: str | None
    if len(y_column_list) == 0:
        y_column = None
    elif len(y_column_list) == 1:
        y_column = y_column_list[0]
    elif x_column is not None:
        y_column = MELTED_Y_COLUMN_NAME
        color_column = MELTED_COLOR_COLUMN_NAME
        columns_to_leave_alone = [x_column]
        if size_column:
            columns_to_leave_alone.append(size_column)
        df = _melt_data(df=df, columns_to_leave_alone=columns_to_leave_alone, columns_to_melt=y_column_list, new_y_column_name=y_column, new_color_column_name=color_column)
    return (df, y_column, color_column)
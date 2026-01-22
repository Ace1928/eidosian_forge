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
def _melt_data(df: pd.DataFrame, columns_to_leave_alone: list[str], columns_to_melt: list[str] | None, new_y_column_name: str, new_color_column_name: str) -> pd.DataFrame:
    """Converts a wide-format dataframe to a long-format dataframe."""
    import pandas as pd
    from pandas.api.types import infer_dtype
    melted_df = pd.melt(df, id_vars=columns_to_leave_alone, value_vars=columns_to_melt, var_name=new_color_column_name, value_name=new_y_column_name)
    y_series = melted_df[new_y_column_name]
    if y_series.dtype == 'object' and 'mixed' in infer_dtype(y_series) and (len(y_series.unique()) > 100):
        raise StreamlitAPIException('The columns used for rendering the chart contain too many values with mixed types. Please select the columns manually via the y parameter.')
    fixed_df = type_util.fix_arrow_incompatible_column_types(melted_df, selected_columns=[*columns_to_leave_alone, new_color_column_name, new_y_column_name])
    return fixed_df
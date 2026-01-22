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
def _get_x_encoding_type(df: pd.DataFrame, chart_type: ChartType, x_column: str | None) -> type_util.VegaLiteType:
    if x_column is None:
        return 'quantitative'
    if chart_type == ChartType.BAR and (not _is_date_column(df, x_column)):
        return 'ordinal'
    return type_util.infer_vegalite_type(df[x_column])
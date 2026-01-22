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
def _get_size_encoding(chart_type: ChartType, size_column: str | None, size_value: str | float | None) -> alt.Size | alt.SizeValue | None:
    import altair as alt
    if chart_type == ChartType.SCATTER:
        if size_column is not None:
            return alt.Size(size_column, legend=SIZE_LEGEND_SETTINGS)
        elif isinstance(size_value, (float, int)):
            return alt.SizeValue(size_value)
        elif size_value is None:
            return alt.SizeValue(100)
        else:
            raise StreamlitAPIException(f'This does not look like a valid size: {repr(size_value)}')
    elif size_column is not None or size_value is not None:
        raise Error(f'Chart type {chart_type.name} does not support size argument. This should never happen!')
    return None
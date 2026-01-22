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
def _get_color_encoding(df: pd.DataFrame, color_value: Color | None, color_column: str | None, y_column_list: list[str], color_from_user: str | Color | list[Color] | None) -> alt.Color | alt.ColorValue | None:
    import altair as alt
    has_color_value = color_value not in [None, [], tuple()]
    if has_color_value:
        if is_color_like(cast(Any, color_value)):
            if len(y_column_list) != 1:
                raise StreamlitColorLengthError([color_value], y_column_list)
            return alt.ColorValue(to_css_color(cast(Any, color_value)))
        elif isinstance(color_value, (list, tuple)):
            color_values = cast(Collection[Color], color_value)
            if len(color_values) != len(y_column_list):
                raise StreamlitColorLengthError(color_values, y_column_list)
            if len(color_value) == 1:
                return alt.ColorValue(to_css_color(cast(Any, color_value[0])))
            else:
                return alt.Color(field=color_column, scale=alt.Scale(range=[to_css_color(c) for c in color_values]), legend=COLOR_LEGEND_SETTINGS, type='nominal', title=' ')
        raise StreamlitInvalidColorError(df, color_from_user)
    elif color_column is not None:
        column_type: str | tuple[str, list[Any]]
        if color_column == MELTED_COLOR_COLUMN_NAME:
            column_type = 'nominal'
        else:
            column_type = type_util.infer_vegalite_type(df[color_column])
        color_enc = alt.Color(field=color_column, legend=COLOR_LEGEND_SETTINGS, type=column_type)
        if color_column == MELTED_COLOR_COLUMN_NAME:
            color_enc['title'] = ' '
        elif len(df[color_column]) and is_color_like(df[color_column].iat[0]):
            color_range = [to_css_color(c) for c in df[color_column].unique()]
            color_enc['scale'] = alt.Scale(range=color_range)
            color_enc['legend'] = None
        else:
            pass
        return color_enc
    return None
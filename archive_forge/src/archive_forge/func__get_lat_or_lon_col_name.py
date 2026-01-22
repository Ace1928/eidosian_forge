from __future__ import annotations
import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any, Collection, Dict, Final, Iterable, Union, cast
from typing_extensions import TypeAlias
import streamlit.elements.deck_gl_json_chart as deck_gl_json_chart
from streamlit import config, type_util
from streamlit.color_util import Color, IntColorTuple, is_color_like, to_int_color_tuple
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DeckGlJsonChart_pb2 import DeckGlJsonChart as DeckGlJsonChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
def _get_lat_or_lon_col_name(data: DataFrame, human_readable_name: str, col_name_from_user: str | None, default_col_names: set[str]) -> str:
    """Returns the column name to be used for latitude or longitude."""
    if isinstance(col_name_from_user, str) and col_name_from_user in data.columns:
        col_name = col_name_from_user
    else:
        candidate_col_name = None
        for c in default_col_names:
            if c in data.columns:
                candidate_col_name = c
                break
        if candidate_col_name is None:
            formatted_allowed_col_name = ', '.join(map(repr, sorted(default_col_names)))
            formmated_col_names = ', '.join(map(repr, list(data.columns)))
            raise StreamlitAPIException(f'Map data must contain a {human_readable_name} column named: {formatted_allowed_col_name}. Existing columns: {formmated_col_names}')
        else:
            col_name = candidate_col_name
    if any(data[col_name].isnull().array):
        raise StreamlitAPIException(f'Column {col_name} is not allowed to contain null values, such as NaN, NaT, or None.')
    return col_name
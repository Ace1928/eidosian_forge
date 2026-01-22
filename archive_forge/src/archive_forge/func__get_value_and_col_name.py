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
def _get_value_and_col_name(data: DataFrame, value_or_name: Any, default_value: Any) -> tuple[Any, str | None]:
    """Take a value_or_name passed in by the Streamlit developer and return a PyDeck
    argument and column name for that property.

    This is used for the size and color properties of the chart.

    Example:
    - If the user passes size=None, this returns the default size value and no column.
    - If the user passes size=42, this returns 42 and no column.
    - If the user passes size="my_col_123", this returns "@@=my_col_123" and "my_col_123".
    """
    pydeck_arg: str | float
    if isinstance(value_or_name, str) and value_or_name in data.columns:
        col_name = value_or_name
        pydeck_arg = f'@@={col_name}'
    else:
        col_name = None
        if value_or_name is None:
            pydeck_arg = default_value
        else:
            pydeck_arg = value_or_name
    return (pydeck_arg, col_name)
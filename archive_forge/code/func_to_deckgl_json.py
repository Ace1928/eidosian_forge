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
def to_deckgl_json(data: Data, lat: str | None, lon: str | None, size: None | str | float, color: None | str | Collection[float], map_style: str | None, zoom: int | None) -> str:
    if data is None:
        return json.dumps(_DEFAULT_MAP)
    if hasattr(data, 'empty') and data.empty:
        return json.dumps(_DEFAULT_MAP)
    df = type_util.convert_anything_to_df(data)
    lat_col_name = _get_lat_or_lon_col_name(df, 'latitude', lat, _DEFAULT_LAT_COL_NAMES)
    lon_col_name = _get_lat_or_lon_col_name(df, 'longitude', lon, _DEFAULT_LON_COL_NAMES)
    size_arg, size_col_name = _get_value_and_col_name(df, size, _DEFAULT_SIZE)
    color_arg, color_col_name = _get_value_and_col_name(df, color, _DEFAULT_COLOR)
    used_columns = sorted([c for c in {lat_col_name, lon_col_name, size_col_name, color_col_name} if c is not None])
    df = df[used_columns]
    color_arg = _convert_color_arg_or_column(df, color_arg, color_col_name)
    zoom, center_lat, center_lon = _get_viewport_details(df, lat_col_name, lon_col_name, zoom)
    default = copy.deepcopy(_DEFAULT_MAP)
    default['initialViewState']['latitude'] = center_lat
    default['initialViewState']['longitude'] = center_lon
    default['initialViewState']['zoom'] = zoom
    default['layers'] = [{'@@type': 'ScatterplotLayer', 'getPosition': f'@@=[{lon_col_name}, {lat_col_name}]', 'getRadius': size_arg, 'radiusMinPixels': 3, 'radiusUnits': 'meters', 'getFillColor': color_arg, 'data': df.to_dict('records')}]
    if map_style:
        if not config.get_option('mapbox.token'):
            raise StreamlitAPIException('You need a Mapbox token in order to select a map type. Refer to the docs for st.map for more information.')
        default['mapStyle'] = map_style
    return json.dumps(default)
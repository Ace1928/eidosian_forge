from __future__ import annotations
import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit import type_util
from streamlit.elements.lib.streamlit_plotly_theme import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics
def _get_embed_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    parsed_embed_url = parsed_url._replace(path=parsed_url.path + '.embed')
    return urllib.parse.urlunparse(parsed_embed_url)
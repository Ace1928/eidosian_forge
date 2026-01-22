from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.utils import get_label_visibility_proto_value
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text
from streamlit.type_util import LabelVisibility, maybe_raise_label_warnings
def _parse_delta(delta: Delta) -> str:
    if delta is None or delta == '':
        return ''
    if isinstance(delta, str):
        return dedent(delta)
    elif isinstance(delta, int) or isinstance(delta, float):
        return str(delta)
    else:
        raise TypeError(f"'{str(delta)}' is of type {str(type(delta))}, which is not an accepted type. delta only accepts: int, float, str, or None. Please convert the value to an accepted type.")
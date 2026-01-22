from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, cast
from streamlit.proto.Json_pb2 import Json as JsonProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state import QueryParamsProxy, SessionStateProxy
from streamlit.user_info import UserInfoProxy
def _ensure_serialization(o: object) -> str | list[Any]:
    """A repr function for json.dumps default arg, which tries to serialize sets as lists"""
    if isinstance(o, set):
        return list(o)
    return repr(o)
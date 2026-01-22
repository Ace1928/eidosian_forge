from __future__ import annotations
import textwrap
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Mapping
from typing_extensions import TypeAlias
from streamlit.errors import DuplicateWidgetID
from streamlit.proto.WidgetStates_pb2 import WidgetState, WidgetStates
from streamlit.runtime.state.common import (
from streamlit.type_util import ValueFieldName
def _build_duplicate_widget_message(widget_func_name: str, user_key: str | None=None) -> str:
    if user_key is not None:
        message = textwrap.dedent("\n            There are multiple widgets with the same `key='{user_key}'`.\n\n            To fix this, please make sure that the `key` argument is unique for each\n            widget you create.\n            ")
    else:
        message = textwrap.dedent("\n            There are multiple identical `st.{widget_type}` widgets with the\n            same generated key.\n\n            When a widget is created, it's assigned an internal key based on\n            its structure. Multiple widgets with an identical structure will\n            result in the same internal key, which causes this error.\n\n            To fix this error, please pass a unique `key` argument to\n            `st.{widget_type}`.\n            ")
    return message.strip('\n').format(widget_type=widget_func_name, user_key=user_key)
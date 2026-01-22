from __future__ import annotations
import ast
import contextlib
import inspect
import re
import types
from typing import TYPE_CHECKING, Any, Final, cast
import streamlit
from streamlit.proto.DocString_pb2 import DocString as DocStringProto
from streamlit.proto.DocString_pb2 import Member as MemberProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_runner import (
from streamlit.runtime.secrets import Secrets
from streamlit.string_util import is_mem_address_str
def _get_docstring(obj):
    doc_string = inspect.getdoc(obj)
    if doc_string is None:
        obj_type = type(obj)
        if obj_type is not type and obj_type is not types.ModuleType and (not inspect.isfunction(obj)) and (not inspect.ismethod(obj)):
            doc_string = inspect.getdoc(obj_type)
    if doc_string:
        return doc_string.strip()
    return None
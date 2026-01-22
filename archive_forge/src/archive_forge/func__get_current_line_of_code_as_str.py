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
def _get_current_line_of_code_as_str():
    scriptrunner_frame = _get_scriptrunner_frame()
    if scriptrunner_frame is None:
        return None
    code_context = scriptrunner_frame.code_context
    if not code_context:
        return None
    code_as_string = ''.join(code_context)
    return re.sub(_NEWLINES, '', code_as_string.strip())
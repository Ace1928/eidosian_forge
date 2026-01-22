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
def _is_computed_property(obj, attr_name):
    obj_class = getattr(obj, '__class__', None)
    if not obj_class:
        return False
    for parent_class in inspect.getmro(obj_class):
        class_attr = getattr(parent_class, attr_name, None)
        if class_attr is None:
            continue
        if isinstance(class_attr, property) or inspect.isgetsetdescriptor(class_attr):
            return True
    return False
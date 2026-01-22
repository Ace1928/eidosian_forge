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
def _is_stcommand(tree, command_name):
    """Checks whether the AST in tree is a call for command_name."""
    root_node = tree.body[0].value
    if not type(root_node) is ast.Call:
        return False
    return getattr(root_node.func, 'id', None) == command_name or getattr(root_node.func, 'attr', None) == command_name
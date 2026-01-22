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
def _get_members(obj):
    members_for_sorting = []
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue
        try:
            is_computed_value = _is_computed_property(obj, attr_name)
            if is_computed_value:
                parent_attr = getattr(obj.__class__, attr_name)
                member_type = 'property'
                weight = 0
                member_docs = _get_docstring(parent_attr)
                member_value = None
            else:
                attr_value = getattr(obj, attr_name)
                weight = _get_weight(attr_value)
                human_readable_value = _get_human_readable_value(attr_value)
                member_type = _get_type_as_str(attr_value)
                if human_readable_value is None:
                    member_docs = _get_docstring(attr_value)
                    member_value = None
                else:
                    member_docs = None
                    member_value = human_readable_value
        except AttributeError:
            continue
        if member_type == 'module':
            continue
        member = MemberProto()
        member.name = attr_name
        member.type = member_type
        if member_docs is not None:
            member.doc_string = _get_first_line(member_docs)
        if member_value is not None:
            member.value = member_value
        members_for_sorting.append((weight, member))
    if members_for_sorting:
        sorted_members = sorted(members_for_sorting, key=lambda x: (x[0], x[1].name))
        return [m for _, m in sorted_members]
    return []
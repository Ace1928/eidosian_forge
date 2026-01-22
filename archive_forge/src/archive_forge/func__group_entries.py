from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
def _group_entries(self, lst, handle_return_values):
    scope_to_grouper = {}
    group_entries = []
    if isinstance(self.value, DAPGrouper):
        new_lst = lst
    else:
        new_lst = []
        get_presentation = self.py_db.variable_presentation.get_presentation
        for attr_name, attr_value, evaluate_name in lst:
            scope = get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values)
            entry = (attr_name, attr_value, evaluate_name)
            if scope:
                presentation = get_presentation(scope)
                if presentation == 'hide':
                    continue
                elif presentation == 'inline':
                    new_lst.append(entry)
                else:
                    if scope not in scope_to_grouper:
                        grouper = DAPGrouper(scope)
                        scope_to_grouper[scope] = grouper
                    else:
                        grouper = scope_to_grouper[scope]
                    grouper.contents_debug_adapter_protocol.append(entry)
            else:
                new_lst.append(entry)
        for scope in DAPGrouper.SCOPES_SORTED:
            grouper = scope_to_grouper.get(scope)
            if grouper is not None:
                group_entries.append((scope, grouper, None))
    return (new_lst, group_entries)
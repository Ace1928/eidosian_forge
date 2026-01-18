from __future__ import (absolute_import, division, print_function)
import sys
import types
import warnings
from sys import intern as _sys_intern
from collections.abc import Mapping, Set
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.native_jinja import NativeJinjaText
def wrap_var(v):
    if v is None or isinstance(v, AnsibleUnsafe):
        return v
    if isinstance(v, Mapping):
        v = _wrap_dict(v)
    elif isinstance(v, Set):
        v = _wrap_set(v)
    elif is_sequence(v):
        v = _wrap_sequence(v)
    elif isinstance(v, NativeJinjaText):
        v = NativeJinjaUnsafeText(v)
    elif isinstance(v, bytes):
        v = AnsibleUnsafeBytes(v)
    elif isinstance(v, str):
        v = AnsibleUnsafeText(v)
    return v
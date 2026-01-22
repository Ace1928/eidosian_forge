from __future__ import (absolute_import, division, print_function)
import sys
import types
import warnings
from sys import intern as _sys_intern
from collections.abc import Mapping, Set
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.utils.native_jinja import NativeJinjaText
def _wrap_sequence(v):
    """Wraps a sequence with unsafe, not meant for strings, primarily
    ``tuple`` and ``list``
    """
    v_type = type(v)
    return v_type((wrap_var(item) for item in v))
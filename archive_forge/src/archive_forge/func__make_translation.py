from __future__ import absolute_import, division, print_function
import os
import platform
import socket
import traceback
import ansible.module_utils.compat.typing as t
from ansible.module_utils.basic import (
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.facts.system.service_mgr import ServiceMgrFactCollector
from ansible.module_utils.facts.utils import get_file_lines, get_file_content
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.six import PY3, text_type
def _make_translation(self, replace_chars, replacement_chars, delete_chars):
    if PY3:
        return str.maketrans(replace_chars, replacement_chars, delete_chars)
    if not isinstance(replace_chars, text_type) or not isinstance(replacement_chars, text_type):
        raise ValueError('replace_chars and replacement_chars must both be strings')
    if len(replace_chars) != len(replacement_chars):
        raise ValueError('replacement_chars must be the same length as replace_chars')
    table = dict(zip((ord(c) for c in replace_chars), replacement_chars))
    for char in delete_chars:
        table[ord(char)] = None
    return table
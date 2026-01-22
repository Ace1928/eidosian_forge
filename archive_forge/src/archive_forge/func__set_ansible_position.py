from __future__ import (absolute_import, division, print_function)
import sys as _sys
from collections.abc import Sequence
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
def _set_ansible_position(self, obj):
    try:
        src, line, col = obj
    except (TypeError, ValueError):
        raise AssertionError('ansible_pos can only be set with a tuple/list of three values: source, line number, column number')
    self._data_source = src
    self._line_number = line
    self._column_number = col
from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def are_lists_equal(s, t):
    if s is None and t is None:
        return True
    if s is None or t is None or len(s) != len(t):
        return False
    if len(s) == 0:
        return True
    s = to_dict(s)
    t = to_dict(t)
    if isinstance(s[0], dict):
        sorted_s = sort_list_of_dictionary(s)
        sorted_t = sort_list_of_dictionary(t)
        for index, d in enumerate(sorted_s):
            if not is_dictionary_subset(d, sorted_t[index]):
                return False
        return True
    else:
        try:
            for elem in s:
                t.remove(elem)
        except ValueError:
            return False
        return not t
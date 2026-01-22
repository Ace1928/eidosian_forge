from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def _recursive_dict_without_none(a_dict, exclude=None):
    """
    Remove all entries with `None` value from a dict, recursively.
    Also drops all entries with keys in `exclude` in the top level.
    """
    if exclude is None:
        exclude = []
    result = {}
    for k, v in a_dict.items():
        if v is not None and k not in exclude:
            if isinstance(v, dict):
                v = _recursive_dict_without_none(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                v = [_recursive_dict_without_none(element) for element in v]
            result[k] = v
    return result
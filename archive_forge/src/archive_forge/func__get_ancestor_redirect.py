from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
def _get_ancestor_redirect(redirected_package_map, fullname):
    cur_pkg = fullname
    while cur_pkg:
        cur_pkg = cur_pkg.rpartition('.')[0]
        ancestor_redirect = redirected_package_map.get(cur_pkg)
        if ancestor_redirect:
            redirect = ancestor_redirect + fullname[len(cur_pkg):]
            return redirect
    return None
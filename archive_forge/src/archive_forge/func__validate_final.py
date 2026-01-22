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
def _validate_final(self):
    if self._split_name[1:3] == ['ansible', 'builtin']:
        self._subpackage_search_paths = []
    elif not self._subpackage_search_paths:
        raise ImportError('no {0} found in {1}'.format(self._package_to_load, self._candidate_paths))
    else:
        self._subpackage_search_paths = [self._subpackage_search_paths[0]]
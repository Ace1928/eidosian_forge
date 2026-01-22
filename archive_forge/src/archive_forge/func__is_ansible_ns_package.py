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
def _is_ansible_ns_package(self, package):
    origin = getattr(package, 'origin', None)
    if not origin:
        return False
    if origin == SYNTHETIC_PACKAGE_NAME:
        return True
    module_filename = os.path.basename(origin)
    return module_filename in {'__synthetic__', '__init__.py'}
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
def _get_collection_name_from_path(path):
    """
    Return the containing collection name for a given path, or None if the path is not below a configured collection, or
    the collection cannot be loaded (eg, the collection is masked by another of the same name higher in the configured
    collection roots).
    :param path: path to evaluate for collection containment
    :return: collection name or None
    """
    path = to_native(os.path.abspath(to_bytes(path)))
    path_parts = path.split('/')
    if path_parts.count('ansible_collections') != 1:
        return None
    ac_pos = path_parts.index('ansible_collections')
    if len(path_parts) < ac_pos + 3:
        return None
    candidate_collection_name = '.'.join(path_parts[ac_pos + 1:ac_pos + 3])
    try:
        imported_pkg_path = to_native(os.path.dirname(to_bytes(import_module('ansible_collections.' + candidate_collection_name).__file__)))
    except ImportError:
        return None
    original_path_prefix = os.path.join('/', *path_parts[0:ac_pos + 3])
    imported_pkg_path = to_native(os.path.abspath(to_bytes(imported_pkg_path)))
    if original_path_prefix != imported_pkg_path:
        return None
    return candidate_collection_name
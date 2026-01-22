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
def _get_collection_metadata(collection_name):
    collection_name = to_native(collection_name)
    if not collection_name or not isinstance(collection_name, string_types) or len(collection_name.split('.')) != 2:
        raise ValueError('collection_name must be a non-empty string of the form namespace.collection')
    try:
        collection_pkg = import_module('ansible_collections.' + collection_name)
    except ImportError:
        raise ValueError('unable to locate collection {0}'.format(collection_name))
    _collection_meta = getattr(collection_pkg, '_collection_meta', None)
    if _collection_meta is None:
        raise ValueError('collection metadata was not loaded for collection {0}'.format(collection_name))
    return _collection_meta
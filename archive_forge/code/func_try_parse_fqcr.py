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
@staticmethod
def try_parse_fqcr(ref, ref_type):
    """
        Attempt to parse a string as a fully-qualified collection reference, returning None on failure (instead of raising an error)
        :param ref: collection reference to parse (a valid ref is of the form 'ns.coll.resource' or 'ns.coll.subdir1.subdir2.resource')
        :param ref_type: the type of the reference, eg 'module', 'role', 'doc_fragment'
        :return: a populated AnsibleCollectionRef object on successful parsing, else None
        """
    try:
        return AnsibleCollectionRef.from_fqcr(ref, ref_type)
    except ValueError:
        pass
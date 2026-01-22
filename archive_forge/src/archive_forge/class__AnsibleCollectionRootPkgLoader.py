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
class _AnsibleCollectionRootPkgLoader(_AnsibleCollectionPkgLoaderBase):

    def _validate_args(self):
        super(_AnsibleCollectionRootPkgLoader, self)._validate_args()
        if len(self._split_name) != 1:
            raise ImportError('this loader can only load the ansible_collections toplevel package, not {0}'.format(self._fullname))
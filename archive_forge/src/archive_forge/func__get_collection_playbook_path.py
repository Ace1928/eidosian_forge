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
def _get_collection_playbook_path(playbook):
    acr = AnsibleCollectionRef.try_parse_fqcr(playbook, u'playbook')
    if acr:
        try:
            pkg = import_module(acr.n_python_collection_package_name)
        except (IOError, ModuleNotFoundError) as e:
            pkg = None
        if pkg:
            cpath = os.path.join(sys.modules[acr.n_python_collection_package_name].__file__.replace('__synthetic__', 'playbooks'))
            if acr.subdirs:
                paths = [to_native(x) for x in acr.subdirs.split(u'.')]
                paths.insert(0, cpath)
                cpath = os.path.join(*paths)
            path = os.path.join(cpath, to_native(acr.resource))
            if os.path.exists(to_bytes(path)):
                return (acr.resource, path, acr.collection)
            elif not acr.resource.endswith(PB_EXTENSIONS):
                for ext in PB_EXTENSIONS:
                    path = os.path.join(cpath, to_native(acr.resource + ext))
                    if os.path.exists(to_bytes(path)):
                        return (acr.resource, path, acr.collection)
    return None
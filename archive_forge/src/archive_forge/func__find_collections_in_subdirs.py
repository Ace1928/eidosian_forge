from __future__ import (absolute_import, division, print_function)
import os
import typing as t
from collections import namedtuple
from collections.abc import MutableSequence, MutableMapping
from glob import iglob
from urllib.parse import urlparse
from yaml import safe_load
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection import HAS_PACKAGING, PkgReq
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
def _find_collections_in_subdirs(dir_path):
    b_dir_path = to_bytes(dir_path, errors='surrogate_or_strict')
    subdir_glob_pattern = os.path.join(b_dir_path, b'*')
    for subdir in iglob(subdir_glob_pattern):
        if os.path.isfile(os.path.join(subdir, _MANIFEST_JSON)):
            yield subdir
        elif os.path.isfile(os.path.join(subdir, _GALAXY_YAML)):
            yield subdir
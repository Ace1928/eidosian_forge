from __future__ import (absolute_import, division, print_function)
import json
import os
import tarfile
import subprocess
import typing as t
from contextlib import contextmanager
from hashlib import sha256
from urllib.error import URLError
from urllib.parse import urldefrag
from shutil import rmtree
from tempfile import mkdtemp
from ansible.errors import AnsibleError
from ansible.galaxy import get_collections_galaxy_meta_info
from ansible.galaxy.api import should_retry_error
from ansible.galaxy.dependency_resolution.dataclasses import _GALAXY_YAML
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.urls import open_url
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
import yaml
def _normalize_galaxy_yml_manifest(galaxy_yml, b_galaxy_yml_path, require_build_metadata=True):
    galaxy_yml_schema = get_collections_galaxy_meta_info()
    mandatory_keys = set()
    string_keys = set()
    list_keys = set()
    dict_keys = set()
    sentinel_keys = set()
    for info in galaxy_yml_schema:
        if info.get('required', False):
            mandatory_keys.add(info['key'])
        key_list_type = {'str': string_keys, 'list': list_keys, 'dict': dict_keys, 'sentinel': sentinel_keys}[info.get('type', 'str')]
        key_list_type.add(info['key'])
    all_keys = frozenset(mandatory_keys | string_keys | list_keys | dict_keys | sentinel_keys)
    set_keys = set(galaxy_yml.keys())
    missing_keys = mandatory_keys.difference(set_keys)
    if missing_keys:
        msg = "The collection galaxy.yml at '%s' is missing the following mandatory keys: %s" % (to_native(b_galaxy_yml_path), ', '.join(sorted(missing_keys)))
        if require_build_metadata:
            raise AnsibleError(msg)
        display.warning(msg)
        raise ValueError(msg)
    extra_keys = set_keys.difference(all_keys)
    if len(extra_keys) > 0:
        display.warning("Found unknown keys in collection galaxy.yml at '%s': %s" % (to_text(b_galaxy_yml_path), ', '.join(extra_keys)))
    for optional_string in string_keys:
        if optional_string not in galaxy_yml:
            galaxy_yml[optional_string] = None
    for optional_list in list_keys:
        list_val = galaxy_yml.get(optional_list, None)
        if list_val is None:
            galaxy_yml[optional_list] = []
        elif not isinstance(list_val, list):
            galaxy_yml[optional_list] = [list_val]
    for optional_dict in dict_keys:
        if optional_dict not in galaxy_yml:
            galaxy_yml[optional_dict] = {}
    for optional_sentinel in sentinel_keys:
        if optional_sentinel not in galaxy_yml:
            galaxy_yml[optional_sentinel] = Sentinel
    if not galaxy_yml.get('version'):
        galaxy_yml['version'] = '*'
    return galaxy_yml
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
def _get_meta_from_src_dir(b_path, require_build_metadata=True):
    galaxy_yml = os.path.join(b_path, _GALAXY_YAML)
    if not os.path.isfile(galaxy_yml):
        raise LookupError("The collection galaxy.yml path '{path!s}' does not exist.".format(path=to_native(galaxy_yml)))
    with open(galaxy_yml, 'rb') as manifest_file_obj:
        try:
            manifest = yaml_load(manifest_file_obj)
        except yaml.error.YAMLError as yaml_err:
            raise AnsibleError("Failed to parse the galaxy.yml at '{path!s}' with the following error:\n{err_txt!s}".format(path=to_native(galaxy_yml), err_txt=to_native(yaml_err))) from yaml_err
    if not isinstance(manifest, dict):
        if require_build_metadata:
            raise AnsibleError(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
        display.warning(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
        raise ValueError(f"The collection galaxy.yml at '{to_native(galaxy_yml)}' is incorrectly formatted.")
    return _normalize_galaxy_yml_manifest(manifest, galaxy_yml, require_build_metadata)
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
def _get_json_from_installed_dir(b_path, filename):
    b_json_filepath = os.path.join(b_path, to_bytes(filename, errors='surrogate_or_strict'))
    try:
        with open(b_json_filepath, 'rb') as manifest_fd:
            b_json_text = manifest_fd.read()
    except (IOError, OSError):
        raise LookupError("The collection {manifest!s} path '{path!s}' does not exist.".format(manifest=filename, path=to_native(b_json_filepath)))
    manifest_txt = to_text(b_json_text, errors='surrogate_or_strict')
    try:
        manifest = json.loads(manifest_txt)
    except ValueError:
        raise AnsibleError('Collection tar file member {member!s} does not contain a valid json string.'.format(member=filename))
    return manifest
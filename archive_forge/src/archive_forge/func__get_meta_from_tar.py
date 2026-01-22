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
def _get_meta_from_tar(b_path):
    if not os.path.exists(b_path):
        raise AnsibleError(f"Unable to find collection artifact file at '{to_native(b_path)}'.")
    if not tarfile.is_tarfile(b_path):
        raise AnsibleError("Collection artifact at '{path!s}' is not a valid tar file.".format(path=to_native(b_path)))
    with tarfile.open(b_path, mode='r') as collection_tar:
        try:
            member = collection_tar.getmember(MANIFEST_FILENAME)
        except KeyError:
            raise AnsibleError("Collection at '{path!s}' does not contain the required file {manifest_file!s}.".format(path=to_native(b_path), manifest_file=MANIFEST_FILENAME))
        with _tarfile_extract(collection_tar, member) as (_member, member_obj):
            if member_obj is None:
                raise AnsibleError('Collection tar file does not contain member {member!s}'.format(member=MANIFEST_FILENAME))
            text_content = to_text(member_obj.read(), errors='surrogate_or_strict')
            try:
                manifest = json.loads(text_content)
            except ValueError:
                raise AnsibleError('Collection tar file member {member!s} does not contain a valid json string.'.format(member=MANIFEST_FILENAME))
            return manifest['collection_info']
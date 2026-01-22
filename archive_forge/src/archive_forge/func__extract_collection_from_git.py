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
def _extract_collection_from_git(repo_url, coll_ver, b_path):
    name, version, git_url, fragment = parse_scm(repo_url, coll_ver)
    b_checkout_path = mkdtemp(dir=b_path, prefix=to_bytes(name, errors='surrogate_or_strict'))
    try:
        git_executable = get_bin_path('git')
    except ValueError as err:
        raise AnsibleError('Could not find git executable to extract the collection from the Git repository `{repo_url!s}`.'.format(repo_url=to_native(git_url))) from err
    if version == 'HEAD':
        git_clone_cmd = (git_executable, 'clone', '--depth=1', git_url, to_text(b_checkout_path))
    else:
        git_clone_cmd = (git_executable, 'clone', git_url, to_text(b_checkout_path))
    try:
        subprocess.check_call(git_clone_cmd)
    except subprocess.CalledProcessError as proc_err:
        raise AnsibleError('Failed to clone a Git repository from `{repo_url!s}`.'.format(repo_url=to_native(git_url))) from proc_err
    git_switch_cmd = (git_executable, 'checkout', to_text(version))
    try:
        subprocess.check_call(git_switch_cmd, cwd=b_checkout_path)
    except subprocess.CalledProcessError as proc_err:
        raise AnsibleError('Failed to switch a cloned Git repo `{repo_url!s}` to the requested revision `{commitish!s}`.'.format(commitish=to_native(version), repo_url=to_native(git_url))) from proc_err
    return os.path.join(b_checkout_path, to_bytes(fragment)) if fragment else b_checkout_path
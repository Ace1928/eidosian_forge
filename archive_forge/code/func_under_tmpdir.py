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
@classmethod
@contextmanager
def under_tmpdir(cls, temp_dir_base, validate_certs=True, keyring=None, required_signature_count=None, ignore_signature_errors=None, require_build_metadata=True):
    """Custom ConcreteArtifactsManager constructor with temp dir.

        This method returns a context manager that allocates and cleans
        up a temporary directory for caching the collection artifacts
        during the dependency resolution process.
        """
    temp_path = mkdtemp(dir=to_bytes(temp_dir_base, errors='surrogate_or_strict'))
    b_temp_path = to_bytes(temp_path, errors='surrogate_or_strict')
    try:
        yield cls(b_temp_path, validate_certs, keyring=keyring, required_signature_count=required_signature_count, ignore_signature_errors=ignore_signature_errors)
    finally:
        rmtree(b_temp_path)
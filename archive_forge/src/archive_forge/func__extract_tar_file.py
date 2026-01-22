from __future__ import (absolute_import, division, print_function)
import errno
import fnmatch
import functools
import json
import os
import pathlib
import queue
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import typing as t
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, fields as dc_fields
from hashlib import sha256
from io import BytesIO
from importlib.metadata import distribution
from itertools import chain
import ansible.constants as C
from ansible.compat.importlib_resources import files
from ansible.errors import AnsibleError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.galaxy_api_proxy import MultiGalaxyAPIProxy
from ansible.galaxy.collection.gpg import (
from ansible.galaxy.dependency_resolution.dataclasses import (
from ansible.galaxy.dependency_resolution.versioning import meets_requirements
from ansible.plugins.loader import get_all_plugin_loaders
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.yaml import yaml_dump
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash, secure_hash_s
from ansible.utils.sentinel import Sentinel
def _extract_tar_file(tar, filename, b_dest, b_temp_path, expected_hash=None):
    """ Extracts a file from a collection tar. """
    with _get_tar_file_member(tar, filename) as (tar_member, tar_obj):
        if tar_member.type == tarfile.SYMTYPE:
            actual_hash = _consume_file(tar_obj)
        else:
            with tempfile.NamedTemporaryFile(dir=b_temp_path, delete=False) as tmpfile_obj:
                actual_hash = _consume_file(tar_obj, tmpfile_obj)
        if expected_hash and actual_hash != expected_hash:
            raise AnsibleError("Checksum mismatch for '%s' inside collection at '%s'" % (to_native(filename, errors='surrogate_or_strict'), to_native(tar.name)))
        b_dest_filepath = os.path.abspath(os.path.join(b_dest, to_bytes(filename, errors='surrogate_or_strict')))
        b_parent_dir = os.path.dirname(b_dest_filepath)
        if not _is_child_path(b_parent_dir, b_dest):
            raise AnsibleError("Cannot extract tar entry '%s' as it will be placed outside the collection directory" % to_native(filename, errors='surrogate_or_strict'))
        if not os.path.exists(b_parent_dir):
            os.makedirs(b_parent_dir, mode=493)
        if tar_member.type == tarfile.SYMTYPE:
            b_link_path = to_bytes(tar_member.linkname, errors='surrogate_or_strict')
            if not _is_child_path(b_link_path, b_dest, link_name=b_dest_filepath):
                raise AnsibleError("Cannot extract symlink '%s' in collection: path points to location outside of collection '%s'" % (to_native(filename), b_link_path))
            os.symlink(b_link_path, b_dest_filepath)
        else:
            shutil.move(to_bytes(tmpfile_obj.name, errors='surrogate_or_strict'), b_dest_filepath)
            tar_member = tar.getmember(to_native(filename, errors='surrogate_or_strict'))
            new_mode = 420
            if stat.S_IMODE(tar_member.mode) & stat.S_IXUSR:
                new_mode |= 73
            os.chmod(b_dest_filepath, new_mode)
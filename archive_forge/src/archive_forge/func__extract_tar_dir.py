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
def _extract_tar_dir(tar, dirname, b_dest):
    """ Extracts a directory from a collection tar. """
    dirname = to_native(dirname, errors='surrogate_or_strict').removesuffix(os.path.sep)
    try:
        tar_member = tar._ansible_normalized_cache[dirname]
    except KeyError:
        raise AnsibleError("Unable to extract '%s' from collection" % dirname)
    b_dir_path = os.path.join(b_dest, to_bytes(dirname, errors='surrogate_or_strict'))
    b_parent_path = os.path.dirname(b_dir_path)
    try:
        os.makedirs(b_parent_path, mode=493)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if tar_member.type == tarfile.SYMTYPE:
        b_link_path = to_bytes(tar_member.linkname, errors='surrogate_or_strict')
        if not _is_child_path(b_link_path, b_dest, link_name=b_dir_path):
            raise AnsibleError("Cannot extract symlink '%s' in collection: path points to location outside of collection '%s'" % (to_native(dirname), b_link_path))
        os.symlink(b_link_path, b_dir_path)
    elif not os.path.isdir(b_dir_path):
        os.mkdir(b_dir_path, 493)
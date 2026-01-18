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
def write_source_metadata(collection, b_collection_path, artifacts_manager):
    source_data = artifacts_manager.get_galaxy_artifact_source_info(collection)
    b_yaml_source_data = to_bytes(yaml_dump(source_data), errors='surrogate_or_strict')
    b_info_dest = collection.construct_galaxy_info_path(b_collection_path)
    b_info_dir = os.path.split(b_info_dest)[0]
    if os.path.exists(b_info_dir):
        shutil.rmtree(b_info_dir)
    try:
        os.mkdir(b_info_dir, mode=493)
        with open(b_info_dest, mode='w+b') as fd:
            fd.write(b_yaml_source_data)
        os.chmod(b_info_dest, 420)
    except Exception:
        if os.path.isdir(b_info_dir):
            shutil.rmtree(b_info_dir)
        raise
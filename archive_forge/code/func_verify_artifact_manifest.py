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
def verify_artifact_manifest(manifest_file, signatures, keyring, required_signature_count, ignore_signature_errors):
    failed_verify = False
    coll_path_parts = to_text(manifest_file, errors='surrogate_or_strict').split(os.path.sep)
    collection_name = '%s.%s' % (coll_path_parts[-3], coll_path_parts[-2])
    if not verify_file_signatures(collection_name, manifest_file, signatures, keyring, required_signature_count, ignore_signature_errors):
        raise AnsibleError(f'Not installing {collection_name} because GnuPG signature verification failed.')
    display.vvvv(f'GnuPG signature verification succeeded for {collection_name}')
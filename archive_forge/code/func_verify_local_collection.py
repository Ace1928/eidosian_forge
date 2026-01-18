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
def verify_local_collection(local_collection, remote_collection, artifacts_manager):
    """Verify integrity of the locally installed collection.

    :param local_collection: Collection being checked.
    :param remote_collection: Upstream collection (optional, if None, only verify local artifact)
    :param artifacts_manager: Artifacts manager.
    :return: a collection verify result object.
    """
    result = CollectionVerifyResult(local_collection.fqcn)
    b_collection_path = to_bytes(local_collection.src, errors='surrogate_or_strict')
    display.display("Verifying '{coll!s}'.".format(coll=local_collection))
    display.display(u"Installed collection found at '{path!s}'".format(path=to_text(local_collection.src)))
    modified_content = []
    verify_local_only = remote_collection is None
    get_json_from_validation_source = functools.partial(_get_json_from_installed_dir, b_collection_path)
    get_hash_from_validation_source = functools.partial(_get_file_hash, b_collection_path)
    if not verify_local_only:
        if local_collection.ver != remote_collection.ver:
            err = "{local_fqcn!s} has the version '{local_ver!s}' but is being compared to '{remote_ver!s}'".format(local_fqcn=local_collection.fqcn, local_ver=local_collection.ver, remote_ver=remote_collection.ver)
            display.display(err)
            result.success = False
            return result
    manifest_file = os.path.join(to_text(b_collection_path, errors='surrogate_or_strict'), MANIFEST_FILENAME)
    signatures = list(local_collection.signatures)
    if verify_local_only and local_collection.source_info is not None:
        signatures = [info['signature'] for info in local_collection.source_info['signatures']] + signatures
    elif not verify_local_only and remote_collection.signatures:
        signatures = list(remote_collection.signatures) + signatures
    keyring_configured = artifacts_manager.keyring is not None
    if not keyring_configured and signatures:
        display.warning('The GnuPG keyring used for collection signature verification was not configured but signatures were provided by the Galaxy server. Configure a keyring for ansible-galaxy to verify the origin of the collection. Skipping signature verification.')
    elif keyring_configured:
        if not verify_file_signatures(local_collection.fqcn, manifest_file, signatures, artifacts_manager.keyring, artifacts_manager.required_successful_signature_count, artifacts_manager.ignore_signature_errors):
            result.success = False
            return result
        display.vvvv(f'GnuPG signature verification succeeded, verifying contents of {local_collection}')
    if verify_local_only:
        manifest_hash = get_hash_from_validation_source(MANIFEST_FILENAME)
    elif keyring_configured and remote_collection.signatures:
        manifest_hash = get_hash_from_validation_source(MANIFEST_FILENAME)
    else:
        b_temp_tar_path = artifacts_manager.get_artifact_path_from_unknown(remote_collection)
        display.vvv(u"Remote collection cached as '{path!s}'".format(path=to_text(b_temp_tar_path)))
        get_json_from_validation_source = functools.partial(_get_json_from_tar_file, b_temp_tar_path)
        get_hash_from_validation_source = functools.partial(_get_tar_file_hash, b_temp_tar_path)
        manifest_hash = get_hash_from_validation_source(MANIFEST_FILENAME)
        _verify_file_hash(b_collection_path, MANIFEST_FILENAME, manifest_hash, modified_content)
    display.display('MANIFEST.json hash: {manifest_hash}'.format(manifest_hash=manifest_hash))
    manifest = get_json_from_validation_source(MANIFEST_FILENAME)
    file_manifest_data = manifest['file_manifest_file']
    file_manifest_filename = file_manifest_data['name']
    expected_hash = file_manifest_data['chksum_%s' % file_manifest_data['chksum_type']]
    _verify_file_hash(b_collection_path, file_manifest_filename, expected_hash, modified_content)
    file_manifest = get_json_from_validation_source(file_manifest_filename)
    collection_dirs = set()
    collection_files = {os.path.join(b_collection_path, b'MANIFEST.json'), os.path.join(b_collection_path, b'FILES.json')}
    for manifest_data in file_manifest['files']:
        name = manifest_data['name']
        if manifest_data['ftype'] == 'file':
            collection_files.add(os.path.join(b_collection_path, to_bytes(name, errors='surrogate_or_strict')))
            expected_hash = manifest_data['chksum_%s' % manifest_data['chksum_type']]
            _verify_file_hash(b_collection_path, name, expected_hash, modified_content)
        if manifest_data['ftype'] == 'dir':
            collection_dirs.add(os.path.join(b_collection_path, to_bytes(name, errors='surrogate_or_strict')))
    for root, dirs, files in os.walk(b_collection_path):
        for name in files:
            full_path = os.path.join(root, name)
            path = to_text(full_path[len(b_collection_path) + 1:], errors='surrogate_or_strict')
            if full_path not in collection_files:
                modified_content.append(ModifiedContent(filename=path, expected='the file does not exist', installed='the file exists'))
        for name in dirs:
            full_path = os.path.join(root, name)
            path = to_text(full_path[len(b_collection_path) + 1:], errors='surrogate_or_strict')
            if full_path not in collection_dirs:
                modified_content.append(ModifiedContent(filename=path, expected='the directory does not exist', installed='the directory exists'))
    if modified_content:
        result.success = False
        display.display('Collection {fqcn!s} contains modified content in the following files:'.format(fqcn=to_text(local_collection.fqcn)))
        for content_change in modified_content:
            display.display('    %s' % content_change.filename)
            display.v('    Expected: %s\n    Found: %s' % (content_change.expected, content_change.installed))
    else:
        what = 'are internally consistent with its manifest' if verify_local_only else 'match the remote collection'
        display.display("Successfully verified that checksums for '{coll!s}' {what!s}.".format(coll=local_collection, what=what))
    return result
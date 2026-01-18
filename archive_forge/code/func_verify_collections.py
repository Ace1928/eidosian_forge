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
def verify_collections(collections, search_paths, apis, ignore_errors, local_verify_only, artifacts_manager):
    """Verify the integrity of locally installed collections.

    :param collections: The collections to check.
    :param search_paths: Locations for the local collection lookup.
    :param apis: A list of GalaxyAPIs to query when searching for a collection.
    :param ignore_errors: Whether to ignore any errors when verifying the collection.
    :param local_verify_only: When True, skip downloads and only verify local manifests.
    :param artifacts_manager: Artifacts manager.
    :return: list of CollectionVerifyResult objects describing the results of each collection verification
    """
    results = []
    api_proxy = MultiGalaxyAPIProxy(apis, artifacts_manager)
    with _display_progress():
        for collection in collections:
            try:
                if collection.is_concrete_artifact:
                    raise AnsibleError(message="'{coll_type!s}' type is not supported. The format namespace.name is expected.".format(coll_type=collection.type))
                default_err = 'Collection %s is not installed in any of the collection paths.' % collection.fqcn
                for search_path in search_paths:
                    b_search_path = to_bytes(os.path.join(search_path, collection.namespace, collection.name), errors='surrogate_or_strict')
                    if not os.path.isdir(b_search_path):
                        continue
                    if not _is_installed_collection_dir(b_search_path):
                        default_err = 'Collection %s does not have a MANIFEST.json. A MANIFEST.json is expected if the collection has been built and installed via ansible-galaxy' % collection.fqcn
                        continue
                    local_collection = Candidate.from_dir_path(b_search_path, artifacts_manager)
                    supplemental_signatures = [get_signature_from_source(source, display) for source in collection.signature_sources or []]
                    local_collection = Candidate(local_collection.fqcn, local_collection.ver, local_collection.src, local_collection.type, signatures=frozenset(supplemental_signatures))
                    break
                else:
                    raise AnsibleError(message=default_err)
                if local_verify_only:
                    remote_collection = None
                else:
                    signatures = api_proxy.get_signatures(local_collection)
                    signatures.extend([get_signature_from_source(source, display) for source in collection.signature_sources or []])
                    remote_collection = Candidate(collection.fqcn, collection.ver if collection.ver != '*' else local_collection.ver, None, 'galaxy', frozenset(signatures))
                    try:
                        if artifacts_manager.keyring is None or not signatures:
                            api_proxy.get_collection_version_metadata(remote_collection)
                    except AnsibleError as e:
                        expected_error_msg = 'Failed to find collection {coll.fqcn!s}:{coll.ver!s}'.format(coll=collection)
                        if e.message == expected_error_msg:
                            raise AnsibleError("Failed to find remote collection '{coll!s}' on any of the galaxy servers".format(coll=collection))
                        raise
                result = verify_local_collection(local_collection, remote_collection, artifacts_manager)
                results.append(result)
            except AnsibleError as err:
                if ignore_errors:
                    display.warning("Failed to verify collection '{coll!s}' but skipping due to --ignore-errors being set. Error: {err!s}".format(coll=collection, err=to_text(err)))
                else:
                    raise
    return results
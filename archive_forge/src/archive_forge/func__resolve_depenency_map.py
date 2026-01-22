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
def _resolve_depenency_map(requested_requirements, galaxy_apis, concrete_artifacts_manager, preferred_candidates, no_deps, allow_pre_release, upgrade, include_signatures, offline):
    """Return the resolved dependency map."""
    if not HAS_RESOLVELIB:
        raise AnsibleError('Failed to import resolvelib, check that a supported version is installed')
    if not HAS_PACKAGING:
        raise AnsibleError('Failed to import packaging, check that a supported version is installed')
    req = None
    try:
        dist = distribution('ansible-core')
    except Exception:
        pass
    else:
        req = next((rr for r in dist.requires or [] if (rr := PkgReq(r)).name == 'resolvelib'), None)
    finally:
        if req is None:
            if not RESOLVELIB_LOWERBOUND <= RESOLVELIB_VERSION < RESOLVELIB_UPPERBOUND:
                raise AnsibleError(f'ansible-galaxy requires resolvelib<{RESOLVELIB_UPPERBOUND.vstring},>={RESOLVELIB_LOWERBOUND.vstring}')
        elif not req.specifier.contains(RESOLVELIB_VERSION.vstring):
            raise AnsibleError(f'ansible-galaxy requires {req.name}{req.specifier}')
    pre_release_hint = '' if allow_pre_release else 'Hint: Pre-releases hosted on Galaxy or Automation Hub are not installed by default unless a specific version is requested. To enable pre-releases globally, use --pre.'
    collection_dep_resolver = build_collection_dependency_resolver(galaxy_apis=galaxy_apis, concrete_artifacts_manager=concrete_artifacts_manager, preferred_candidates=preferred_candidates, with_deps=not no_deps, with_pre_releases=allow_pre_release, upgrade=upgrade, include_signatures=include_signatures, offline=offline)
    try:
        return collection_dep_resolver.resolve(requested_requirements, max_rounds=2000000).mapping
    except CollectionDependencyResolutionImpossible as dep_exc:
        conflict_causes = ('* {req.fqcn!s}:{req.ver!s} ({dep_origin!s})'.format(req=req_inf.requirement, dep_origin='direct request' if req_inf.parent is None else 'dependency of {parent!s}'.format(parent=req_inf.parent)) for req_inf in dep_exc.causes)
        error_msg_lines = list(chain(('Failed to resolve the requested dependencies map. Could not satisfy the following requirements:',), conflict_causes))
        error_msg_lines.append(pre_release_hint)
        raise AnsibleError('\n'.join(error_msg_lines)) from dep_exc
    except CollectionDependencyInconsistentCandidate as dep_exc:
        parents = ['%s.%s:%s' % (p.namespace, p.name, p.ver) for p in dep_exc.criterion.iter_parent() if p is not None]
        error_msg_lines = ["Failed to resolve the requested dependencies map. Got the candidate {req.fqcn!s}:{req.ver!s} ({dep_origin!s}) which didn't satisfy all of the following requirements:".format(req=dep_exc.candidate, dep_origin='direct request' if not parents else 'dependency of {parent!s}'.format(parent=', '.join(parents)))]
        for req in dep_exc.criterion.iter_requirement():
            error_msg_lines.append('* {req.fqcn!s}:{req.ver!s}'.format(req=req))
        error_msg_lines.append(pre_release_hint)
        raise AnsibleError('\n'.join(error_msg_lines)) from dep_exc
    except ValueError as exc:
        raise AnsibleError(to_native(exc)) from exc